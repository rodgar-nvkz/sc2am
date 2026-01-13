"""
Training script for Marine vs 2 Zerglings RL environment using TorchRL PPO.

This uses TorchRL's native PPO implementation with torch.compile and CudaGraphs
for high performance training without monkey-patching.

Usage:
    python -m scam.train.ppo_torchrl train --steps 100000 --num-envs 8
    python -m scam.train.ppo_torchrl train --steps 100000 --num-envs 8 --compile --cudagraphs
    python -m scam.train.ppo_torchrl eval --games 10
    python -m scam.train.ppo_torchrl train --steps 100000 --resume artifacts/models/marine_torchrl_ppo_xxx.pt
"""

import argparse
import glob
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import EnvCreator, ParallelEnv, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.modules import MLP, OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from scam.envs.trl_v0 import NUM_ACTIONS, OBS_SIZE, SC2TRLEnv
from scam.settings import PROJECT_ROOT

# torch.set_float32_matmul_precision("high")


def make_env(upgrade_levels=[2, ]):
    """Create a single SC2TorchRLEnv with transforms."""
    env = TransformedEnv(SC2TRLEnv({"upgrade_level": upgrade_levels, "game_steps_per_env": [2]}, device="cpu"))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env


def make_parallel_env(num_envs: int):
    return ParallelEnv(num_workers=num_envs, create_env_fn=EnvCreator(make_env), shared_memory=True, mp_start_method="fork")


def make_ppo_models(device: torch.device):
    actor_mlp = MLP(
        in_features=OBS_SIZE,
        out_features=NUM_ACTIONS,
        num_cells=[48, 32],
        activation_class=nn.Tanh,
        device=device,
    )

    for layer in actor_mlp.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

    actor_module = TensorDictModule(
        actor_mlp,
        in_keys=["observation"],
        out_keys=["logits"],
    )

    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    critic_mlp = MLP(
        in_features=OBS_SIZE,
        out_features=1,
        num_cells=[48, 32],
        activation_class=nn.Tanh,
        device=device,
    )

    for layer in critic_mlp.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)

    critic = ValueOperator(
        module=critic_mlp,
        in_keys=["observation"],
    )

    return actor, critic


def train(steps: int, seed: int, num_envs: int, resume: str | None = None):
    """Train a PPO agent using TorchRL"""
    # Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device setup
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Hyperparameters

    frames_per_batch = 1024 * num_envs  # n_steps * num_envs
    total_frames = math.ceil(steps / frames_per_batch) * frames_per_batch
    mini_batch_size = 256
    num_epochs = 4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.05
    critic_coef = 0.5
    max_grad_norm = 0.5
    lr = 1e-4

    num_mini_batches = frames_per_batch // mini_batch_size

    print(f"Launching {num_envs} parallel environments...")
    env = make_parallel_env(num_envs)

    actor, critic = make_ppo_models(device)
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        # compile_policy={"mode": "default", "warmup": 5},
        # cudagraph_policy={"warmup": 10},
    )

    # Create replay buffer
    sampler = SamplerWithoutReplacement()
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=sampler,
        batch_size=mini_batch_size,
    )

    adv_module = GAE(
        gamma=gamma,
        lmbda=gae_lambda,
        value_network=critic,
        average_gae=False,
        device=device,
    )

    # Create loss module
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_coeff=entropy_coef,
        critic_coeff=critic_coef,
        normalize_advantage=True,
        loss_critic_type="smooth_l1",
    )
    loss_module.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=lr, eps=1e-5)

    # Learning rate scheduler (linear annealing like SB3)
    total_batches = steps // frames_per_batch
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=total_batches,
    )

    if resume:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])

    # Training loop
    start_time = time.time()
    episode_rewards = deque(maxlen=100)
    episode_wins = deque(maxlen=100)
    episode_count, collected_frames = 0, 0

    print(f"\nStarting training for {steps} frames...")
    print(f"Frames per batch: {frames_per_batch}, Mini-batch size: {mini_batch_size}")
    print(f"Epochs per batch: {num_epochs}, Mini-batches per epoch: {num_mini_batches}")

    for i, data in enumerate(collector):
        collected_frames += data.numel()

        # Extract episode statistics from completed episodes (using TorchRL's episode_reward from RewardSum)
        done_mask = data["next", "done"].squeeze(-1)
        if done_mask.any():
            # Get final episode rewards for completed episodes
            ep_rewards = data["next", "episode_reward"][done_mask].cpu().numpy().flatten()
            num_done = len(ep_rewards)
            episode_count += num_done
            for r in ep_rewards:
                episode_rewards.append(float(r))
                episode_wins.append(float(r) > 0)  # Win if positive reward

        # Compute GAE
        with torch.no_grad():
            data = adv_module(data)

        # Flatten data for mini-batch sampling
        data_flat = data.reshape(-1)

        # Extend buffer ONCE before all epochs (not inside epoch loop)
        replay_buffer.empty()
        replay_buffer.extend(data_flat)

        # PPO update epochs
        loss = None
        loss_dict = None
        for epoch in range(num_epochs):
            for _ in range(num_mini_batches):
                batch = replay_buffer.sample()

                # Compute loss
                loss_dict = loss_module(batch)
                loss = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optimizer.step()

        # Step learning rate scheduler after each batch
        lr_scheduler.step()

        # Logging
        elapsed = time.time() - start_time
        fps = collected_frames / elapsed if elapsed > 0 else 0

        # Log every iteration (metrics averaged over last 100 episodes, matching vanilla PPO)
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        win_rate = np.mean(episode_wins) * 100 if episode_wins else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        log_line = (
            f"Batch {i+1} | "
            f"F: {collected_frames:,} | "
            f"FPS: {fps:.0f} | "
            f"Ep: {episode_count} | "
            f"R: {avg_reward:.2f} | "
            f"Win: {win_rate:.1f}% | "
            f"LR: {current_lr:.2e} | "
        )
        # PPO diagnostics from last update (using TorchRL's computed values)
        if loss_dict is not None and loss is not None:
            explained_var = loss_dict["explained_variance"].mean().item() if "explained_variance" in loss_dict.keys() else 0.0
            log_line += (
                f"Loss: {loss.item():.4f} | "
                f"Clip: {loss_dict['clip_fraction'].item():.6f} | "
                f"KL: {loss_dict['kl_approx'].item():.4f} | "
                f"Ent: {loss_dict['entropy'].item():.3f} | "
                f"ExpVar: {explained_var:.3f}"
            )
        print(log_line)

    collector.shutdown()

    # Save model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "collected_frames": collected_frames,
        "episode_count": episode_count,
    }

    model_path = model_dir / f"marine_torchrl_ppo_{timestamp}.pt"
    torch.save(checkpoint, model_path)
    print(f"\nModel saved to {model_path}")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")

    return str(model_path)


def eval(num_games: int = 10, model_path: str | None = None):
    device = torch.device("cpu")

    model_path = model_path or max(glob.glob("artifacts/models/marine_torchrl_ppo_*.pt"), key=os.path.getctime)
    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    actor, _ = make_ppo_models(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    # Create eval environment
    env = make_env(upgrade_levels=[1])

    wins = 0
    total_rewards = []

    print(f"\nEvaluating for {num_games} games...")

    for game in range(num_games):
        td = env.reset()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                td = actor(td)

            td = env.step(td)
            done = td["next", "done"].item()
            episode_reward += td["next", "reward"].item()
            td = step_mdp(td)

        wins += episode_reward > 0
        total_rewards.append(episode_reward)
        print(f"Game {game + 1}: reward={episode_reward:.2f}, won={episode_reward > 0}")

    replay_data = env.game.clients[0].save_replay()
    replay_dir = PROJECT_ROOT / "artifacts" / "replays" / Path(model_path).stem
    replay_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    replay_path = replay_dir / f"{timestamp}.SC2Replay"
    replay_path.write_bytes(replay_data)

    env.close()

    print(f"\n=== Results over {num_games} games ===")
    print(f"Win rate: {wins}/{num_games} ({100 * wins / num_games:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate Marine vs 2 Zerglings with TorchRL PPO")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument("--steps", type=int, default=100_000, help="Total training frames")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    eval_parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")

    args = parser.parse_args()

    if args.command == "train":
        train(steps=args.steps, seed=args.seed, num_envs=args.envs, resume=args.resume)
    elif args.command == "eval":
        eval(num_games=args.games, model_path=args.model)
    else:
        parser.print_help()
