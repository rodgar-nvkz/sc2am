"""
IMPALA-style training for SC2 Marine vs Zerglings environment.

This implements fully autonomous workers with:
- Local policy copies for inference (no per-step IPC)
- Async weight synchronization from main process
- V-trace off-policy correction
- Rollout-based communication

Hybrid Action Space (Autoregressive):
- Discrete commands: STAY, MOVE, ATTACK_Z1, ATTACK_Z2
- Continuous angle (sin, cos) for MOVE command
- Angle head is conditioned on the discrete command via embedding

Architecture:
    Worker 1: [SC2 + Policy] -> rollouts -> Queue ->
    Worker 2: [SC2 + Policy] -> rollouts -> Queue ->  Main: V-trace -> Train -> Broadcast weights
    Worker N: [SC2 + Policy] -> rollouts -> Queue ->
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
from scam.impala_v2.collector import RolloutBatch, collector_worker
from scam.impala_v2.config import IMPALAConfig
from scam.impala_v2.eval import eval_model
from scam.impala_v2.interop import SharedWeights
from scam.impala_v2.model import ActorCritic
from scam.train.impala import compute_vtrace


def train(total_frames: int, num_workers: int, seed: int = 42, resume: str | None = None):
    """Main training loop."""
    mp.set_start_method("spawn")
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    config = IMPALAConfig(total_frames=total_frames, num_workers=num_workers, upgrade_levels=[1])
    model = ActorCritic(OBS_SIZE, NUM_COMMANDS).to(device)
    if resume:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    shared_weights = SharedWeights(model)
    rollout_queue = Queue(maxsize=num_workers * 2)
    shutdown_event = Event()

    # Start workers
    workers = []
    for i in range(num_workers):
        args = i, rollout_queue, shared_weights, shutdown_event, config
        p = Process(target=collector_worker, args=args, daemon=True)
        p.start()
        workers.append(p)

    print(f"Started {num_workers} workers")

    # LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-4)
    total_updates = total_frames // (config.rollout_length * num_workers)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, config.lr_start_factor, config.lr_end_factor, total_updates)

    # Tracking
    update_count = 0
    collected_frames = 0
    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    start_time = time.time()

    print(f"\nStarting IMPALA training for {total_frames:,} frames...")
    print(f"Rollout length: {config.rollout_length}, Workers: {num_workers}")
    print(f"Frames per update: {config.rollout_length * num_workers:,}")

    try:
        while collected_frames < total_frames:
            rollout_batch = RolloutBatch.create(num_workers, config.rollout_length, OBS_SIZE, NUM_COMMANDS)

            while not rollout_batch.is_full():
                try:
                    rollout = rollout_queue.get(timeout=60.0)
                    rollout_batch.insert(rollout)

                except Exception as e:
                    logger.warning(f"Timeout waiting for rollouts: {e}")
                    alive_workers = sum(1 for w in workers if w.is_alive())
                    if alive_workers == 0:
                        raise RuntimeError("All workers have died!") from e
                    continue

            # Track episode statistics
            episode_returns += rollout_batch.episode_returns
            episode_lengths += rollout_batch.episode_lengths
            collected_frames += rollout_batch._count * config.rollout_length
            batch = rollout_batch.to_tensors(device)

            # Compute V-trace targets with current policy
            model.eval()
            with torch.no_grad():
                # Get current policy log probs and values with action masking
                _, _, target_cmd_log_probs, target_angle_log_probs, _, current_values = model.get_action_and_value(
                    batch["observations"],
                    batch["commands"],
                    batch["angles"],
                    batch["action_masks"],
                )
                bootstrap_values = model.get_value(batch["next_observations"])

                # Combined log probs for importance sampling
                # For V-trace, we use the combined log prob
                target_log_probs = target_cmd_log_probs + target_angle_log_probs
                behavior_log_probs = batch["behavior_cmd_log_probs"] + batch["behavior_angle_log_probs"]

            vtrace_targets, advantages = compute_vtrace(
                behavior_log_probs=behavior_log_probs,
                target_log_probs=target_log_probs,
                rewards=batch["rewards"],
                values=batch["behavior_values"],  # Use behavior values for TD errors
                bootstrap_value=bootstrap_values,
                dones=batch["dones"],
                gamma=config.gamma,
                rho_bar=config.rho_bar,
                c_bar=config.c_bar,
            )

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Flatten for mini-batch training
            B, T = batch["observations"].shape[:2]
            flat_obs = batch["observations"].reshape(B * T, -1)
            flat_commands = batch["commands"].reshape(B * T)
            flat_angles = batch["angles"].reshape(B * T, 2)
            flat_advantages = advantages.reshape(B * T)
            flat_vtrace_targets = vtrace_targets.reshape(B * T)
            flat_old_cmd_log_probs = batch["behavior_cmd_log_probs"].reshape(B * T)
            flat_old_angle_log_probs = batch["behavior_angle_log_probs"].reshape(B * T)
            flat_action_masks = batch["action_masks"].reshape(B * T, -1)

            # PPO-style mini-batch updates
            model.train()
            indices = np.arange(B * T)

            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            num_batches = 0

            for _ in range(config.num_epochs):
                np.random.shuffle(indices)

                for start in range(0, B * T, config.mini_batch_size):
                    end = start + config.mini_batch_size
                    mb_indices = indices[start:end]

                    mb_obs = flat_obs[mb_indices]
                    mb_commands = flat_commands[mb_indices]
                    mb_angles = flat_angles[mb_indices]
                    mb_advantages = flat_advantages[mb_indices]
                    mb_vtrace_targets = flat_vtrace_targets[mb_indices]
                    mb_old_cmd_log_probs = flat_old_cmd_log_probs[mb_indices]
                    mb_old_angle_log_probs = flat_old_angle_log_probs[mb_indices]
                    mb_action_masks = flat_action_masks[mb_indices]

                    # Forward pass (with action masking)
                    _, _, new_cmd_log_probs, new_angle_log_probs, entropy, new_values = model.get_action_and_value(
                        mb_obs, mb_commands, mb_angles, mb_action_masks
                    )

                    # === Command Policy Loss (standard PPO) ===
                    cmd_ratio = torch.exp(new_cmd_log_probs - mb_old_cmd_log_probs)
                    cmd_surr1 = cmd_ratio * mb_advantages
                    cmd_surr2 = torch.clamp(cmd_ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * mb_advantages
                    cmd_policy_loss = -torch.min(cmd_surr1, cmd_surr2).mean()

                    # === Angle Policy Loss (masked by P(MOVE)) ===
                    # Sparse but correct gradients. It's fine since MOVE is common action.
                    move_mask = (mb_commands == ACTION_MOVE).float()
                    angle_ratio = torch.exp(new_angle_log_probs - mb_old_angle_log_probs)
                    angle_surr1 = angle_ratio * mb_advantages * move_mask
                    angle_surr2 = torch.clamp(angle_ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * mb_advantages * move_mask
                    # Normalize by number of MOVE actions to avoid magnitude issues
                    num_moves = move_mask.sum().clamp(min=1.0)
                    angle_policy_loss = -torch.min(angle_surr1, angle_surr2).sum() / num_moves

                    # Combined policy loss
                    policy_loss = cmd_policy_loss + angle_policy_loss
                    value_loss = F.smooth_l1_loss(new_values, mb_vtrace_targets)
                    entropy_loss = -entropy.mean()
                    loss = policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    num_batches += 1

            update_count += 1
            lr_scheduler.step()
            shared_weights.push(model)

            # Logging
            elapsed = time.time() - start_time
            fps = collected_frames / elapsed if elapsed > 0 else 0
            avg_return = np.mean(episode_returns) if episode_returns else 0.0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0.0  # noqa: F841
            win_rate = np.mean([r > 0 for r in episode_returns]) * 100 if episode_returns else 0.0

            # Weight staleness (how old are the weights workers are using)
            avg_staleness = np.mean(shared_weights.get_version() - rollout_batch.weight_versions)

            print(
                f"Update {update_count:4d} | "
                f"Frames: {collected_frames:>10,} | "
                f"FPS: {fps:>6.0f} | "
                f"AVG Ret: {avg_return:>3.2f} | "
                f"Win: {win_rate:>5.1f}% | "
                f"Loss: {total_loss/num_batches:>6.3f} | "
                f"Entropy: {total_entropy/num_batches:>5.2f} | "
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
        "collected_frames": collected_frames,
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
    train_parser.add_argument("--steps", type=int, default=1_000_000, help="Total training frames")
    train_parser.add_argument("--num-workers", type=int, default=10, help="Number of worker processes")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    eval_parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")

    args = parser.parse_args()

    if args.command == "train":
        train(total_frames=args.steps, num_workers=args.num_workers, seed=args.seed, resume=args.resume)
    elif args.command == "eval":
        eval_model(num_games=args.games, model_path=args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
