"""
IMPALA-style training for SC2 Marine vs Zerglings environment.

Data architecture: 1 game = 1 episode = 1 training unit
- Workers collect complete episodes
- Learner batches N episodes, computes loss on all steps at once
- Single forward pass, single backward pass

Key optimizations:
- Batched episode processing (pad + stack)
- Single forward pass for all episodes
- torch.compile for kernel fusion
- GPU-optimized tensor operations
- CPU thread limiting to prevent oversubscription
- Lock-free weight sharing between processes

Architecture:
    Worker 1: [SC2 + Policy] -> episode -> Queue ->
    Worker 2: [SC2 + Policy] -> episode -> Queue ->  Main: V-trace -> Train -> Broadcast weights
    Worker N: [SC2 + Policy] -> episode -> Queue ->
                                              ^
                                    Shared memory weights
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import os
import sys
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
from .env import SC2GymEnv
from .eval import eval_model
from .interop import SharedWeights
from .model import ActorCritic, ModelConfig


def train(total_episodes: int, num_workers: int, seed: int = 42, resume: str | None = None):
    """Main training loop with batched GPU processing."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    # Main process can use more threads since it's doing GPU work mostly (workers will set their own)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    mp.set_start_method("spawn")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model config from environment class constants (required, no defaults)
    model_config = ModelConfig(
        obs_size=SC2GymEnv.OBS_SIZE,
        num_commands=SC2GymEnv.NUM_COMMANDS,
        move_action_id=SC2GymEnv.MOVE_ACTION_ID,
    )
    config = IMPALAConfig(
        model=model_config,
        total_episodes=total_episodes,
        num_workers=num_workers,
        upgrade_levels=[1],
    )

    model = ActorCritic(config.model).to(device)

    if resume:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    compiled_model: Any = torch.compile(model, mode="default", fullgraph=False)

    episode_queue = Queue(maxsize=num_workers * 2)
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
    episode_lengths: deque[int] = deque(maxlen=100)
    episode_returns: deque[float] = deque(maxlen=100)
    start_time = time.time()

    print(f"\nStarting IMPALA training for {total_episodes:,} episodes...")
    print(f"Episodes per batch: {config.episodes_per_batch}, Workers: {num_workers}")
    print(f"Model total parameters: {sum(p.numel() for p in model.parameters()):,}")

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

            # Convert to tensors
            tensors = batch.to_tensors(device)

            # Normalize advantages - only scale by std, do NOT center, ruins angle head!
            advantages = tensors["advantages"]
            advantages = advantages / (advantages.std() + 1e-8)

            # Precompute move mask for angle loss
            move_mask = (tensors["commands"] == config.model.move_action_id).float()

            # === Training loop: multiple epochs over the batch ===
            compiled_model.train()
            total_loss = 0.0
            total_entropy = 0.0

            for _ in range(config.num_epochs):
                # Forward pass through model
                output = compiled_model(
                    obs=tensors["observations"],
                    command=tensors["commands"],
                    angle=tensors["angles"],
                    action_mask=tensors["action_masks"],
                )

                # Compute losses using encapsulated head methods
                losses = model.compute_losses(
                    output=output,
                    old_cmd_log_prob=tensors["behavior_cmd_log_probs"],
                    old_angle_log_prob=tensors["behavior_angle_log_probs"],
                    advantages=advantages,
                    vtrace_targets=tensors["vtrace_targets"],
                    move_mask=move_mask,
                    clip_epsilon=config.clip_epsilon,
                )

                # Combine losses
                cmd_loss = losses["command"].loss
                angle_loss = losses["angle"].loss
                value_loss = losses["value"].loss

                # Masked entropy: only count angle entropy for MOVE commands
                entropy = output.total_entropy(move_mask).mean()

                policy_loss = cmd_loss + angle_loss
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                total_entropy += entropy.item()

            update_count += 1
            lr_scheduler.step()
            shared_weights.push(model)  # Always push uncompiled model weights

            # Average over epochs for logging
            avg_loss = total_loss / config.num_epochs
            avg_entropy = total_entropy / config.num_epochs

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
                f"Loss: {avg_loss:>6.3f} | "
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

    model_path = model_dir / f"smarte_v3_{timestamp}.pt"
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

    return str(model_path)


# ============================================================================
# CLI
# ============================================================================

logger.remove()
logger.add(sys.stderr, level="INFO")
torch.set_float32_matmul_precision("high")


def main():
    parser = argparse.ArgumentParser(description="IMPALA training with hybrid action space for Marine vs Zerglings")
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
        eval_model(num_games=args.games, model_path=args.model, upgrade_level=1)
        eval_model(num_games=args.games, model_path=args.model, upgrade_level=0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
