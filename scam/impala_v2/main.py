"""
IMPALA-style training for SC2 Marine vs Zerglings environment.

Data architecture: 1 game = 1 episode = 1 training unit
- Workers collect complete episodes
- Learner batches N episodes, computes loss on all steps at once
- Single forward pass, single backward pass

Architecture:
    Worker 1: [SC2 + Policy] -> episode -> Queue ->
    Worker 2: [SC2 + Policy] -> episode -> Queue ->  Main: V-trace -> Train -> Broadcast weights
    Worker N: [SC2 + Policy] -> episode -> Queue ->
                                              ^
                                    Shared memory weights
"""

import argparse
import multiprocessing as mp
import time
from collections import deque
from multiprocessing import Event, Process, Queue
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from scam.envs.impala_v2 import ACTION_MOVE, NUM_COMMANDS, OBS_SIZE
from scam.impala_v2.collector import EpisodeBatch, collector_worker
from scam.impala_v2.config import IMPALAConfig
from scam.impala_v2.eval import eval_model
from scam.impala_v2.interop import SharedWeights
from scam.impala_v2.model import ActorCritic


def train(total_episodes: int, num_workers: int, seed: int = 42, resume: str | None = None):
    """Main training loop."""
    mp.set_start_method("spawn")
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    config = IMPALAConfig(
        total_episodes=total_episodes,
        num_workers=num_workers,
        episodes_per_batch=num_workers,
        upgrade_levels=[1],
    )

    model = ActorCritic(OBS_SIZE, NUM_COMMANDS).to(device)
    if resume:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

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
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, config.lr_start_factor, config.lr_end_factor, total_updates)

    # Tracking
    update_count = 0
    collected_episodes = 0
    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    start_time = time.time()

    print(f"\nStarting IMPALA training for {total_episodes:,} episodes...")
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
            batch = EpisodeBatch.from_episodes(episodes, gamma=config.gamma)

            # Track statistics
            episode_returns.extend(batch.episode_returns)
            episode_lengths.extend(batch.episode_lengths)
            collected_episodes += batch.num_episodes

            # Convert to tensors
            tensors = batch.to_tensors(device)

            # Normalize advantages
            advantages = tensors["advantages"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # === Single forward pass on ALL steps ===
            model.train()
            _, _, new_cmd_log_probs, new_angle_log_probs, entropy, new_values = model.get_action_and_value(
                tensors["observations"],
                tensors["commands"],
                tensors["angles"],
                tensors["action_masks"],
            )

            # === Command Policy Loss (PPO-style clipping) ===
            old_cmd_log_probs = tensors["behavior_cmd_log_probs"]
            cmd_ratio = torch.exp(new_cmd_log_probs - old_cmd_log_probs)
            cmd_surr1 = cmd_ratio * advantages
            cmd_surr2 = torch.clamp(cmd_ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
            cmd_policy_loss = -torch.min(cmd_surr1, cmd_surr2).mean()

            # === Angle Policy Loss (masked by MOVE commands) ===
            old_angle_log_probs = tensors["behavior_angle_log_probs"]
            move_mask = (tensors["commands"] == ACTION_MOVE).float()
            angle_ratio = torch.exp(new_angle_log_probs - old_angle_log_probs)
            angle_surr1 = angle_ratio * advantages * move_mask
            angle_surr2 = (
                torch.clamp(angle_ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                * advantages
                * move_mask
            )
            num_moves = move_mask.sum().clamp(min=1.0)
            angle_policy_loss = -torch.min(angle_surr1, angle_surr2).sum() / num_moves

            # === Value Loss ===
            value_loss = F.smooth_l1_loss(new_values, tensors["vtrace_targets"])

            # === Entropy Loss ===
            entropy_loss = -entropy.mean()

            # === Total Loss ===
            policy_loss = cmd_policy_loss + angle_policy_loss
            loss = policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss

            # === Single backward pass ===
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            update_count += 1
            lr_scheduler.step()
            shared_weights.push(model)

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
                f"Loss: {loss.item():>6.3f} | "
                f"Ent: {entropy.mean().item():>5.2f} | "
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
        "collected_episodes": collected_episodes,
        "update_count": update_count,
        "config": config.__dict__,
    }

    model_path = model_dir / f"impala_v2_{timestamp}.pt"
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

    return str(model_path)


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="IMPALA training with hybrid action space for Marine vs Zerglings")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument("--episodes", type=int, default=10_000, help="Total training episodes")
    train_parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    eval_parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")

    args = parser.parse_args()

    if args.command == "train":
        train(total_episodes=args.episodes, num_workers=args.num_workers, seed=args.seed, resume=args.resume)
    elif args.command == "eval":
        eval_model(num_games=args.games, model_path=args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
