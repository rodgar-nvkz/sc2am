"""
IMPALA-style training with LSTM for SC2 Marine vs Zerglings.

Architecture:
    obs → VectorEncoder → LSTM → [ActorHead, CriticHead]

Key optimizations:
- Batched episode processing (pad + stack)
- Single forward pass for all episodes
- torch.compile for kernel fusion
- GPU-optimized tensor operations

Data flow:
    Worker 1: [SC2 + Policy + LSTM] -> episode -> Queue ->
    Worker 2: [SC2 + Policy + LSTM] -> episode -> Queue ->  Main: V-trace -> Train -> Broadcast
    Worker N: [SC2 + Policy + LSTM] -> episode -> Queue ->
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import time
from collections import deque
from multiprocessing import Event, Process, Queue
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from .collector import EpisodeBatch, collector_worker
from .config import IMPALAConfig
from .env import OBS_SIZE
from .eval import eval_model
from .interop import SharedWeights
from .model import ActorCritic


def train(
    total_episodes: int,
    num_workers: int,
    seed: int = 42,
    resume: str | None = None,
    compile_model: bool = True,
):
    """Main training loop with batched GPU processing."""
    mp.set_start_method("spawn")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = IMPALAConfig(total_episodes=total_episodes, num_workers=num_workers, upgrade_levels=[1])
    config.model.obs_size = OBS_SIZE
    # num_actions is computed in __post_init__ from num_move_directions (default=4)

    model = ActorCritic(config.model).to(device)
    compiled_model: Any = model  # For type checker compatibility with torch.compile

    if resume:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Compile model for faster execution (PyTorch 2.0+)
    if compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        # Use reduce-overhead mode for smaller batches
        compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("Model compiled successfully")

    # Print model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Action space: {config.model.num_actions} discrete actions")
    print(f"  - Move directions: 0-{config.model.num_move_directions - 1} ({config.model.num_move_directions} directions)")
    print(f"  - Attack Z1: {config.model.action_attack_z1}, Attack Z2: {config.model.action_attack_z2}")
    print(f"  - Stop: {config.model.action_stop}, Skip: {config.model.action_skip}")

    episode_queue: Queue[Any] = Queue(maxsize=num_workers * 2)
    shared_weights = SharedWeights(model)  # Always use uncompiled model for weight sharing
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
    episode_returns: deque[float] = deque(maxlen=100)
    episode_lengths: deque[int] = deque(maxlen=100)
    episode_wins: deque[bool] = deque(maxlen=100)
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
            episode_wins.extend(batch.episode_wins)
            collected_episodes += batch.num_episodes

            # Transfer batch to GPU once (single transfer for all data)
            tensors = batch.to_tensors(device)
            obs_batch = tensors["observations"]  # (B, T_max, obs_size)
            actions_batch = tensors["actions"]  # (B, T_max)
            old_log_probs_batch = tensors["behavior_log_probs"]  # (B, T_max)
            vtrace_targets_batch = tensors["vtrace_targets"]  # (B, T_max)
            advantages_batch = tensors["advantages"]  # (B, T_max)
            mask_batch = tensors["mask"]  # (B, T_max) - True for valid positions

            # === Training loop: multiple epochs over the batch ===
            compiled_model.train()
            total_loss = 0.0
            total_entropy = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0

            for _ in range(config.num_epochs):
                # Single forward pass for ALL episodes at once
                # LSTM processes (B, T_max, obs_size) in parallel
                output = compiled_model.forward_sequence(
                    obs_seq=obs_batch,
                    hidden=None,  # Start with zeros for each episode
                    actions=actions_batch,
                )

                # output.action.log_prob: (B, T_max)
                # output.action.entropy: (B, T_max)
                # output.value: (B, T_max)

                # Flatten and apply mask to get only valid positions
                mask_flat = mask_batch.reshape(-1)  # (B * T_max,)
                log_probs_flat = output.action.log_prob.reshape(-1)[mask_flat]  # (N_valid,)
                entropies_flat = output.action.entropy.reshape(-1)[mask_flat]  # (N_valid,)
                values_flat = output.value.reshape(-1)[mask_flat]  # (N_valid,)

                old_log_probs_flat = old_log_probs_batch.reshape(-1)[mask_flat]
                advantages_flat = advantages_batch.reshape(-1)[mask_flat]
                vtrace_targets_flat = vtrace_targets_batch.reshape(-1)[mask_flat]

                # Normalize advantages
                advantages_norm = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

                # Compute policy loss (PPO-style clipping)
                ratio = torch.exp(log_probs_flat - old_log_probs_flat)
                surr1 = ratio * advantages_norm
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages_norm
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = torch.nn.functional.smooth_l1_loss(values_flat, vtrace_targets_flat)

                # Entropy bonus
                entropy = entropies_flat.mean()

                # Total loss
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)  # Use uncompiled for params
                optimizer.step()

                total_loss += loss.item()
                total_entropy += entropy.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

            update_count += 1
            lr_scheduler.step()
            shared_weights.push(model)  # Always push uncompiled model weights

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
            win_rate = np.mean(episode_wins) * 100 if episode_wins else 0.0

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

    # Save model (always use uncompiled model)
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
    train_parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument("-g", "--games", type=int, default=10, help="Number of games to play")
    eval_parser.add_argument("-m", "--model", type=str, default=None, help="Path to model checkpoint")

    args = parser.parse_args()

    if args.command == "train":
        train(
            total_episodes=args.episodes,
            num_workers=args.workers,
            seed=args.seed,
            resume=args.resume,
            compile_model=not args.no_compile,
        )
    elif args.command == "eval":
        eval_model(num_games=args.games, model_path=args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
