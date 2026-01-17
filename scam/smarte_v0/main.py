"""
IMPALA-style training with LSTM for SC2 Marine vs Zerglings.

Architecture:
    obs → VectorEncoder → LSTM → [ActorHead, CriticHead]

Key changes from previous version:
- LSTM for sequential memory
- Single discrete action space (40 actions)
- Process episodes sequentially through LSTM during training

Data flow:
    Worker 1: [SC2 + Policy + LSTM] -> episode -> Queue ->
    Worker 2: [SC2 + Policy + LSTM] -> episode -> Queue ->  Main: V-trace -> Train -> Broadcast
    Worker N: [SC2 + Policy + LSTM] -> episode -> Queue ->
"""

import argparse
import dataclasses
import multiprocessing as mp
import time
from collections import deque
from multiprocessing import Event, Process, Queue
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from .collector import EpisodeBatch, collector_worker
from .config import IMPALAConfig
from .env import NUM_ACTIONS, OBS_SIZE
from .eval import eval_model
from .interop import SharedWeights
from .model import ActorCritic


def train(total_episodes: int, num_workers: int, seed: int = 42, resume: str | None = None):
    """Main training loop with LSTM support."""
    mp.set_start_method("spawn")
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    config = IMPALAConfig(total_episodes=total_episodes, num_workers=num_workers, upgrade_levels=[1])
    config.model.obs_size = OBS_SIZE
    config.model.num_actions = NUM_ACTIONS

    model = ActorCritic(config.model).to(device)

    if resume:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Action space: {config.model.num_actions} discrete actions")
    print("  - Move directions: 0-35 (36 x 10°)")
    print("  - Attack Z1: 36, Attack Z2: 37")
    print("  - Stop: 38, Skip: 39")

    episode_queue: Queue = Queue(maxsize=num_workers * 2)
    shared_weights = SharedWeights(model)
    shutdown_event = Event()

    # Start workers
    workers = []
    for i in range(num_workers):
        args = (i, episode_queue, shared_weights, shutdown_event, config)
        p = Process(target=collector_worker, args=args, daemon=True)
        p.start()
        workers.append(p)

    print(f"Started {num_workers} workers")

    # Optimizer and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-4)
    total_updates = total_episodes // config.episodes_per_batch
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, config.lr_start_factor, config.lr_end_factor, total_updates
    )

    # Tracking
    update_count = 0
    collected_episodes = 0
    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    start_time = time.time()

    print(f"\nStarting LSTM IMPALA training for {total_episodes:,} episodes...")
    print(f"Episodes per batch: {config.episodes_per_batch}, Workers: {num_workers}")

    try:
        while collected_episodes < total_episodes:
            # Collect episodes from workers
            episodes = []
            while len(episodes) < config.episodes_per_batch:
                try:
                    episode = episode_queue.get(timeout=60.0)
                    episodes.append(episode)
                except Exception as e:
                    logger.warning(f"Timeout waiting for episodes: {e}")
                    alive_workers = sum(1 for w in workers if w.is_alive())
                    if alive_workers == 0:
                        raise RuntimeError("All workers have died!") from e
                    continue

            # Create batch with V-trace computed per episode
            batch = EpisodeBatch.from_episodes(episodes, config)

            # Track statistics
            episode_returns.extend(batch.episode_returns)
            episode_lengths.extend(batch.episode_lengths)
            collected_episodes += batch.num_episodes

            # === Training loop: multiple epochs over the batch ===
            model.train()
            total_loss = 0.0
            total_entropy = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0

            for _ in range(config.num_epochs):
                # Process each episode through LSTM and accumulate gradients
                all_log_probs = []
                all_entropies = []
                all_values = []
                all_old_log_probs = []
                all_advantages = []
                all_vtrace_targets = []

                for ep_idx, episode in enumerate(batch.episodes):
                    # Convert episode data to tensors
                    obs_seq = torch.from_numpy(episode.observations).unsqueeze(0).to(device)  # (1, T, obs_size)
                    actions = torch.from_numpy(episode.actions).unsqueeze(0).to(device)  # (1, T)

                    # Forward pass through LSTM for entire episode
                    output = model.forward_sequence(
                        obs_seq=obs_seq,
                        hidden=None,  # Start with zeros for each episode
                        actions=actions,
                    )

                    # Collect outputs (squeeze batch dimension)
                    all_log_probs.append(output.action.log_prob.squeeze(0))  # (T,)
                    all_entropies.append(output.action.entropy.squeeze(0))  # (T,)
                    all_values.append(output.value.squeeze(0))  # (T,)

                    # Get pre-computed targets
                    all_old_log_probs.append(
                        torch.from_numpy(episode.behavior_log_probs).to(device)
                    )
                    all_advantages.append(
                        torch.from_numpy(batch.advantages[ep_idx]).to(device)
                    )
                    all_vtrace_targets.append(
                        torch.from_numpy(batch.vtrace_targets[ep_idx]).to(device)
                    )

                # Concatenate all episodes
                log_probs = torch.cat(all_log_probs)
                entropies = torch.cat(all_entropies)
                values = torch.cat(all_values)
                old_log_probs = torch.cat(all_old_log_probs)
                advantages = torch.cat(all_advantages)
                vtrace_targets = torch.cat(all_vtrace_targets)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Compute policy loss (PPO-style clipping)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = torch.nn.functional.smooth_l1_loss(values, vtrace_targets)

                # Entropy bonus
                entropy = entropies.mean()

                # Total loss
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                total_entropy += entropy.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

            update_count += 1
            lr_scheduler.step()
            shared_weights.push(model)

            # Average over epochs for logging
            avg_loss = total_loss / config.num_epochs
            avg_entropy = total_entropy / config.num_epochs
            avg_policy_loss = total_policy_loss / config.num_epochs
            avg_value_loss = total_value_loss / config.num_epochs

            # Logging
            elapsed = time.time() - start_time
            eps_per_sec = collected_episodes / elapsed if elapsed > 0 else 0
            avg_return = np.mean(episode_returns) if episode_returns else 0.0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
            win_rate = np.mean([r > 0 for r in episode_returns]) * 100 if episode_returns else 0.0

            # Weight staleness
            avg_staleness = np.mean(shared_weights.get_version() - np.array(batch.weight_versions))

            print(
                f"Update {update_count:4d} | "
                f"Eps: {collected_episodes:>6,}/{total_episodes:,} | "
                f"E/s: {eps_per_sec:>5.1f} | "
                f"Rew: {avg_return:>5.2f} | "
                f"Len: {avg_length:>5.0f} | "
                f"Win: {win_rate:>5.1f}% | "
                f"Loss: {avg_loss:>6.3f} (π:{avg_policy_loss:.3f} v:{avg_value_loss:.3f}) | "
                f"Ent: {avg_entropy:>5.2f} | "
                f"Stale: {avg_staleness:>4.1f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        print("\nTraining complete!")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Shutdown workers
        print("Shutting down workers...")
        shutdown_event.set()

        # Give workers time to finish
        time.sleep(2.0)

        # Terminate any remaining workers
        for w in workers:
            if w.is_alive():
                w.terminate()
                w.join(timeout=1.0)

    # Save model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": dataclasses.asdict(config.model),
        "collected_episodes": collected_episodes,
        "update_count": update_count,
        "config": dataclasses.asdict(config),
    }

    model_path = model_dir / f"lstm_v0_{timestamp}.pt"
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

    return str(model_path)


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LSTM IMPALA training for Marine vs Zerglings (40 discrete actions)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument("-e", "--episodes", type=int, default=10_000, help="Total training episodes")
    train_parser.add_argument("-w", "--workers", type=int, default=10, help="Number of worker processes")
    train_parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("-r", "--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument("-g", "--games", type=int, default=10, help="Number of games to play")
    eval_parser.add_argument("-m", "--model", type=str, default=None, help="Path to model checkpoint")

    args = parser.parse_args()

    if args.command == "train":
        train(total_episodes=args.episodes, num_workers=args.workers, seed=args.seed, resume=args.resume)
    elif args.command == "eval":
        eval_model(num_games=args.games, model_path=args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
