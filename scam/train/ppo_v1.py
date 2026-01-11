"""
Training script for Marine vs 2 Zerglings RL environment using PPO.

This uses the SC2GymEnv wrapper to convert ParallelEnv to Gymnasium format,
enabling compatibility with Stable Baselines3 while maintaining the ParallelEnv
structure for future multi-agent expansion.

Usage:
    python -m scam.train.ppo_v1 train --steps 100000 --num-envs 10
    python -m scam.train.ppo_v1 eval --games 10
"""

import argparse
import glob
import os
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.ppo import MlpPolicy

from scam.envs.gym_v1 import SC2GymEnv


def make_env_fn(env_id: int):
    """Factory function that returns a function to create an env (avoids lambda closure issues)."""

    def _init():
        return SC2GymEnv()

    return _init


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


def train(steps: int = 100_000, seed: int = 42, num_envs: int = 10, resume: str | None = None):
    """Train a PPO agent to control the Marine.

    Args:
        steps: Total training timesteps
        seed: Random seed
        num_envs: Number of parallel environments
        resume: Optional model filename to resume training from (searches in artifacts/models/)
    """
    print(f"Launching {num_envs} parallel environments...")

    env = SubprocVecEnv([SC2GymEnv for _ in range(num_envs)])
    env = VecMonitor(env)

    print(f"Launched {num_envs} parallel SC2 environments")

    if resume:
        model = PPO.load(resume, env=env, device="cpu")
        logger.info(f"Resumed training from model: {resume}")
    else:
        # Create new model
        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cpu",
            tensorboard_log="./artifacts/logs/marine_vs_2zerglings/",
            policy_kwargs={"net_arch": [32, 32]},
        )
        logger.info("Initialized new PPO model")

    # Train the model with episode logging callback
    callback = EpisodeLoggerCallback(verbose=1)
    model.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=resume is None)

    # Print final stats
    if callback.episode_wins:
        print("\n=== Training Complete ===")
        print(f"Total episodes: {callback.episode_count}")
        print(
            f"Win rate: {sum(callback.episode_wins) / len(callback.episode_wins) * 100:.1f}%"
        )

    # Save the final model
    model_name = f"artifacts/models/marine_v1_ppo_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(model_name)
    print(f"Model saved as {model_name}")

    env.close()
    return model_name


def eval(num_games: int = 10, model_path: str | None = None):
    """Evaluate a trained agent and save replays of best/worst games."""
    env = SC2GymEnv({"game_steps_per_env": 1})

    # Find latest model if not specified
    if model_path is None:
        try:
            model_path = max(
                glob.glob("artifacts/models/marine_v1_ppo_*.zip"), key=os.path.getctime
            )
            print(f"Using latest model: {model_path}")
        except ValueError:
            print("No trained model found. Train one first with: train_marine.py train")
            return

    model = PPO.load(model_path, device="cpu")

    # Extract model name from path (remove .zip extension)
    model_name = Path(model_path).stem

    wins = 0
    total_rewards = []

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
    """Run a quick demo to test the environment works."""
    import numpy as np

    env = SC2GymEnv()
    print("Environment: SC2GymEnv (Marine vs 2 Zerglings)")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    for _ in range(2):
        obs, info = env.reset(seed=42)
        print(f"\nInitial observation: {obs}")

        # Run a few random steps
        for step in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(
                f"Step {step + 1}: action={action}, reward={reward:.3f}, obs={np.round(obs, 2)}"
            )

            if terminated or truncated:
                print(f"Episode ended. Won: {info.get('won', False)}")
                break

    env.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/evaluate Marine vs 2 Zerglings RL agent"
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
        "--resume", type=str, default=None,
        help="Model filename to resume training from (searches in artifacts/models/)"
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument(
        "--games", type=int, default=10, help="Number of games to play"
    )
    eval_parser.add_argument(
        "--model", type=str, default=None, help="Path to model file"
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a quick demo")

    args = parser.parse_args()

    if args.command == "train":
        train(steps=args.steps, seed=args.seed, num_envs=args.envs, resume=args.resume)
    elif args.command == "eval":
        eval(num_games=args.games, model_path=args.model)
    elif args.command == "demo":
        demo()
    else:
        parser.print_help()
