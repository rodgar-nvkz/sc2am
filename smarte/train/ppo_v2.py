"""
Training script for Marine vs 2 Zerglings RL environment using PPO (v2).

This version uses a custom feature extractor to handle Dict observation space
with both unit features (MLP) and terrain grid (CNN).

Usage:
    python -m smarte.train.ppo_v2 train --steps 100000 --num-envs 10
    python -m smarte.train.ppo_v2 eval --games 10
    python -m smarte.train.ppo_v2 demo
"""

import argparse
import glob
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from smarte.envs.gym_v2 import SC2GymEnv


class CombinedFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observation space.

    Processes:
    - unit_features: through MLP
    - terrain: through CNN

    Then concatenates the outputs.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64):
        # We need to compute the total features dimension
        super().__init__(observation_space, features_dim)

        # Get individual space shapes
        unit_features_shape = observation_space["unit_features"].shape[0]  # 14
        terrain_shape = observation_space["terrain"].shape  # (32, 32, 3)

        # MLP for unit features
        self.unit_mlp = nn.Sequential(
            nn.Linear(unit_features_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        unit_features_out = 32

        # CNN for terrain grid
        # Input: (batch, 3, 32, 32) - need to permute from (batch, 32, 32, 3)
        self.terrain_cnn = nn.Sequential(
            nn.Conv2d(
                terrain_shape[2], 16, kernel_size=3, stride=2, padding=1
            ),  # -> (16, 16, 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 4, 4)
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate CNN output size: 32 channels * 4 * 4 = 512
        terrain_features_out = 32 * 4 * 4

        # Combined features dimension
        combined_dim = unit_features_out + terrain_features_out

        # Final projection to features_dim
        self.combined_linear = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process unit features through MLP
        unit_features = self.unit_mlp(observations["unit_features"])

        # Process terrain through CNN
        # Need to permute from (batch, H, W, C) to (batch, C, H, W)
        terrain = observations["terrain"].permute(0, 3, 1, 2)
        terrain_features = self.terrain_cnn(terrain)

        # Concatenate and project
        combined = torch.cat([unit_features, terrain_features], dim=1)
        return self.combined_linear(combined)


class EpisodeLoggerCallback(BaseCallback):
    """Callback to log episode statistics during training."""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_wins: list[bool] = []
        self.episode_count: int = 0

    def _on_step(self) -> bool:
        # Check for episode completions in all envs
        infos = self.locals.get("infos", [])
        for info in infos:
            # VecMonitor adds "episode" key when episode ends
            if "episode" in info:
                self.episode_count += 1
                # Track wins from our custom info
                won = info.get("won", False)
                self.episode_wins.append(won)

                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]

                # Log every 100 episodes
                if self.episode_count % 100 == 0:
                    recent_wins = self.episode_wins[-100:]
                    win_rate = sum(recent_wins) / len(recent_wins) * 100
                    print(
                        f"Episodes: {self.episode_count} | "
                        f"Last Ep Reward: {ep_reward:.2f} | "
                        f"Last Ep Length: {ep_length} | "
                        f"Win Rate (last 100): {win_rate:.1f}%"
                    )
        return True


def make_env_fn(env_id: int, upgrade_level: int | None = None):
    """Factory function that returns a function to create an env."""

    def _init():
        return SC2GymEnv({"upgrade_level": upgrade_level})

    return _init


def train(
    steps: int = 100_000,
    seed: int = 42,
    num_envs: int = 10,
    resume: str | None = None,
    upgrade_level: int | None = None,
):
    """Train a PPO agent to control the Marine.

    Args:
        steps: Total training timesteps
        seed: Random seed
        num_envs: Number of parallel environments
        resume: Optional model filename to resume training from
        upgrade_level: Fixed upgrade level (0-2) or None for domain randomization
    """
    print(f"Launching {num_envs} parallel environments...")
    print(
        f"Upgrade level: {upgrade_level if upgrade_level is not None else 'randomized (0-2)'}"
    )

    # Create environments with optional fixed upgrade level
    env_fns = [make_env_fn(i, upgrade_level) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    print(f"Launched {num_envs} parallel SC2 environments")

    policy_kwargs = {
        "features_extractor_class": CombinedFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 64},
        "net_arch": [64, 64],  # Additional layers after feature extraction
    }

    if resume:
        model = PPO.load(resume, env=env, device="cpu")
        logger.info(f"Resumed training from model: {resume}")
    else:
        # Create new model
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            # device="cpu",
            tensorboard_log="./artifacts/logs/marine_vs_2zerglings_v2/",
            policy_kwargs=policy_kwargs,
        )
        logger.info("Initialized new PPO model with CNN+MLP policy")

    # Train the model with episode logging callback
    callback = EpisodeLoggerCallback(verbose=1)
    model.learn(
        total_timesteps=steps, callback=callback, reset_num_timesteps=resume is None
    )

    # Print final stats
    if callback.episode_wins:
        print("\n=== Training Complete ===")
        print(f"Total episodes: {callback.episode_count}")
        print(
            f"Win rate: {sum(callback.episode_wins) / len(callback.episode_wins) * 100:.1f}%"
        )

    # Save the final model
    model_name = f"artifacts/models/marine_v2_ppo_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(model_name)
    print(f"Model saved as {model_name}")

    env.close()
    return model_name


def eval(
    num_games: int = 10, model_path: str | None = None, upgrade_level: int | None = None
):
    """Evaluate a trained agent and save replays."""
    env = SC2GymEnv()

    # Find latest model if not specified
    if model_path is None:
        try:
            model_path = max(
                glob.glob("artifacts/models/marine_v2_ppo_*.zip"), key=os.path.getctime
            )
            print(f"Using latest model: {model_path}")
        except ValueError:
            print(
                "No trained model found. Train one first with: python -m smarte.train.ppo_v2 train"
            )
            return

    model = PPO.load(model_path, device="cpu")

    # Extract model name from path (remove .zip extension)
    model_name = Path(model_path).stem

    wins = 0
    total_rewards = []

    print(
        f"Evaluating with upgrade_level={upgrade_level if upgrade_level is not None else 'random'}"
    )

    for game in range(num_games):
        obs, info = env.reset(seed=game)
        episode_reward = 0
        done = False
        won = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            done = terminated or truncated

            if done:
                won = info.get("won", False)
                if won:
                    wins += 1

        total_rewards.append(episode_reward)
        print(f"Game {game + 1}: reward={episode_reward:.2f}, won={won}")

    # Save replay
    replay_data = env.game.clients[0].save_replay()
    project_root = Path(__file__).parent.parent.parent
    replay_dir = project_root / "artifacts" / "replays" / model_name
    replay_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    replay_path = replay_dir / f"{timestamp}.SC2Replay"
    replay_path.write_bytes(replay_data)

    env.close()

    print(f"\n=== Results over {num_games} games ===")
    print(f"Win rate: {wins}/{num_games} ({100 * wins / num_games:.1f}%)")
    print(f"Average reward: {sum(total_rewards) / len(total_rewards):.2f}")


def demo():
    """Run a quick demo to test the environment and observation space."""
    env = SC2GymEnv()
    print("Environment: SC2GymEnv v2 (Marine vs 2 Zerglings with Dict obs)")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    for ep in range(2):
        print(f"\n=== Episode {ep + 1} ===")
        obs, info = env.reset(seed=42 + ep)
        print(f"Unit features shape: {obs['unit_features'].shape}")
        print(f"Terrain shape: {obs['terrain'].shape}")
        print(f"Unit features: {np.round(obs['unit_features'], 2)}")
        print(f"Terrain walkable cells: {obs['terrain'][:, :, 0].sum():.0f}")
        print(
            f"Allied positions (non-zero): {np.count_nonzero(obs['terrain'][:, :, 1])}"
        )
        print(
            f"Enemy positions (non-zero): {np.count_nonzero(obs['terrain'][:, :, 2])}"
        )

        # Run a few random steps
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 10 == 0:
                print(
                    f"Step {step + 1}: action={action}, reward={reward:.3f}, "
                    f"allied_hp={obs['unit_features'][0]:.2f}"
                )

            if terminated or truncated:
                print(
                    f"Episode ended at step {step + 1}. Won: {info.get('won', False)}"
                )
                break

    env.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/evaluate Marine vs 2 Zerglings RL agent (v2 with terrain)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument(
        "--steps", type=int, default=100_000, help="Total training steps"
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument(
        "--envs", type=int, default=8, help="Number of parallel environments"
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Model filename to resume training from",
    )
    train_parser.add_argument(
        "--upgrade-level",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Fixed upgrade level (0-2). If not set, randomizes each episode.",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument(
        "--games", type=int, default=10, help="Number of games to play"
    )
    eval_parser.add_argument(
        "--model", type=str, default=None, help="Path to model file"
    )
    eval_parser.add_argument(
        "--upgrade-level",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Fixed upgrade level for evaluation",
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a quick demo")

    args = parser.parse_args()

    if args.command == "train":
        train(
            steps=args.steps,
            seed=args.seed,
            num_envs=args.envs,
            resume=args.resume,
            upgrade_level=args.upgrade_level,
        )
    elif args.command == "eval":
        eval(
            num_games=args.games,
            model_path=args.model,
            upgrade_level=args.upgrade_level,
        )
    elif args.command == "demo":
        demo()
    else:
        parser.print_help()
