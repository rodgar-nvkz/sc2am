"""
Training script for SC2 environments using DreamerV3.

Uses the SC2GymEnv single-agent wrapper for RLlib compatibility.
DreamerV3 is a single-agent algorithm, so we train one marine agent.

Usage:
    python -m scam.train.dv3 train --num-envs 2 --steps 100000
    python -m scam.train.dv3 train --model-size S --training-ratio 512
    python -m scam.train.dv3 eval --checkpoint <path> --games 10
"""

import argparse
import glob
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
from ray.rllib.core.rl_module import RLModule
from ray.tune.registry import register_env

from scam.envs.gym import SC2GymEnv

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.evaluation.episode_v2 import EpisodeV2


class MetricsCallback(DefaultCallbacks):
    """Custom callback to track win rate and training throughput."""

    def __init__(self):
        super().__init__()
        self._last_timestep = 0
        self._last_time = None
        self._episode_wins: list[bool] = []
        self._episode_rewards: list[float] = []
        self._episode_count = 0

    def on_episode_end(
        self,
        *,
        episode,
        env_runner: "EnvRunner" = None,
        metrics_logger=None,
        env=None,
        env_index: int,
        rl_module=None,
        **kwargs,
    ) -> None:
        """Called at the end of each episode to track win rate."""
        # Get episode info - handle both old and new API
        if hasattr(episode, "get_infos"):
            infos = episode.get_infos()
            if infos:
                last_info = infos[-1] if isinstance(infos, list) else infos
                won = last_info.get("won", False)
            else:
                won = False
        elif hasattr(episode, "last_info_for"):
            info = episode.last_info_for("default_agent")
            won = info.get("won", False) if info else False
        else:
            won = False

        self._episode_wins.append(won)
        self._episode_count += 1

        # Get episode reward
        if hasattr(episode, "get_return"):
            reward = episode.get_return()
        elif hasattr(episode, "total_reward"):
            reward = episode.total_reward
        else:
            reward = 0.0
        self._episode_rewards.append(reward)

        # Log to metrics logger if available
        if metrics_logger is not None:
            metrics_logger.log_value("win", float(won), window=100)
            metrics_logger.log_value("episode_reward", reward, window=100)

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict,
        **kwargs,
    ) -> None:
        """Called at the end of each training iteration."""
        current_time = time.time()
        current_timestep = result.get("num_env_steps_sampled_lifetime", 0)

        # Calculate steps per second
        if self._last_time is not None and current_timestep > self._last_timestep:
            elapsed = current_time - self._last_time
            steps_delta = current_timestep - self._last_timestep
            steps_per_sec = steps_delta / elapsed if elapsed > 0 else 0
        else:
            steps_per_sec = 0.0

        # Calculate win rate
        if self._episode_wins:
            recent_wins = self._episode_wins[-100:]
            win_rate = sum(recent_wins) / len(recent_wins) * 100
        else:
            win_rate = 0.0

        # Calculate average reward
        if self._episode_rewards:
            recent_rewards = self._episode_rewards[-100:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
        else:
            avg_reward = 0.0

        # Add custom metrics to result
        result["custom_metrics"] = result.get("custom_metrics", {})
        result["custom_metrics"]["steps_per_second"] = steps_per_sec
        result["custom_metrics"]["total_steps"] = current_timestep
        result["custom_metrics"]["win_rate_pct"] = win_rate
        result["custom_metrics"]["avg_reward_last_100"] = avg_reward
        result["custom_metrics"]["total_episodes"] = self._episode_count

        # Print to console for visibility
        print(
            f"[Metrics] Iter {result.get('training_iteration', '?')}: "
            f"{current_timestep:,} steps, "
            f"{self._episode_count} episodes, "
            f"{steps_per_sec:.1f} steps/sec, "
            f"win rate: {win_rate:.1f}%, "
            f"avg reward: {avg_reward:.2f}"
        )

        self._last_time = current_time
        self._last_timestep = current_timestep


def create_env(env_config: dict):
    """
    Environment creator for RLlib.

    Creates a SC2GymEnv single-agent environment.

    Args:
        env_config: Configuration dict (currently unused)

    Returns:
        SC2GymEnv instance
    """
    return SC2GymEnv()


def train(
    num_envs: int = 1,
    steps: int = 100_000,
    model_size: str = "XS",
    training_ratio: int = 32,
    checkpoint_freq: int = 10,
    seed: int = 42,
):
    """
    Train a DreamerV3 agent on the SC2 environment.

    Args:
        num_envs: Number of SC2 game instances (each with N agents)
        steps: Total environment steps to train
        model_size: DreamerV3 model size ("XS", "S", "M", "L", "XL")
        training_ratio: Ratio of trained steps to env steps (default 32 for fast envs, paper uses 512)
        checkpoint_freq: Save checkpoint every N iterations
        seed: Random seed
    """
    import gymnasium as gym

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the environment
    register_env("sc2_dreamer", create_env)

    # Get spaces from the env class without creating an actual instance
    # (avoids starting SC2 in the main process)
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
    act_space = gym.spaces.Discrete(10)

    print(f"=== DreamerV3 Training ===")
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    print(f"Number of env runners: {num_envs}")
    print(f"Model size: {model_size}")
    print(f"Training ratio: {training_ratio} (paper uses 512, lower = faster but needs more env steps)")
    print(f"Total steps: {steps}")

    # Configure DreamerV3
    config = (
        DreamerV3Config()
        .environment(
            env="sc2_dreamer",
            env_config={},
        )
        .training(
            model_size=model_size,
            training_ratio=training_ratio,
            # Batch settings - smaller for faster iteration
            batch_size_B=8,
            batch_length_T=32,
            # Horizon for actor-critic training in dreams
            horizon_H=15,
            # Learning rates - slightly higher for faster learning
            world_model_lr=3e-4,
            actor_lr=1e-4,
            critic_lr=1e-4,
            # Discount and GAE
            gamma=0.99,
            gae_lambda=0.95,
        )
        .env_runners(
            # Use remote env runners for parallel SC2 instances
            # Each runner spawns its own SC2 process
            num_env_runners=num_envs,
            # Number of envs per runner (1 SC2 game per runner)
            num_envs_per_env_runner=1,
            # Rollout settings
            rollout_fragment_length=1,
            # Don't create env on local worker (main process) - only in remote runners
            create_env_on_local_worker=False,
        )
        .learners(
            # Use 1 GPU learner for training
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .reporting(
            # Report metrics frequently for short episodes
            metrics_num_episodes_for_smoothing=10,
            # Report every iteration for better visibility
            min_sample_timesteps_per_iteration=1,
        )
        .callbacks(MetricsCallback)
        .debugging(
            seed=seed,
        )
    )

    # Create results directory (must be absolute path for Ray/PyArrow)
    results_dir = Path("artifacts/results/dreamerv3").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run training with Tune
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"sc2_dreamerv3_{timestamp}"

    print(f"\nStarting training run: {experiment_name}")
    print(f"Results will be saved to: {results_dir / experiment_name}")

    tuner = tune.Tuner(
        "DreamerV3",
        param_space=config,
        run_config=tune.RunConfig(
            name=experiment_name,
            storage_path=str(results_dir),
            stop={"num_env_steps_sampled_lifetime": steps},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=checkpoint_freq,
                checkpoint_at_end=True,
            ),
            verbose=1,
        ),
    )

    results = tuner.fit()

    # Print results
    best_result = results.get_best_result()
    print(f"\n=== Training Complete ===")
    print(f"Best result: {best_result}")

    # Print final checkpoint path for easy eval
    if best_result.checkpoint:
        print(f"\nTo evaluate this model, run:")
        print(f"  python -m scam.train.dv3 eval --checkpoint {best_result.checkpoint.path}")

    ray.shutdown()


def eval(
    checkpoint_path: str | None = None,
    num_games: int = 10,
):
    """
    Evaluate a trained DreamerV3 agent.

    Args:
        checkpoint_path: Path to checkpoint directory. If None, uses latest.
        num_games: Number of games to play for evaluation.
    """
    # Find checkpoint if not specified
    if checkpoint_path is None:
        # Look for latest checkpoint in results directory
        results_dir = Path("artifacts/results/dreamerv3").resolve()
        checkpoint_dirs = glob.glob(str(results_dir / "**/checkpoint_*"), recursive=True)
        if not checkpoint_dirs:
            print("No checkpoints found. Train a model first with: python -m scam.train.dv3 train")
            return
        checkpoint_path = max(checkpoint_dirs, key=os.path.getctime)
        print(f"Using latest checkpoint: {checkpoint_path}")

    # Find the RLModule checkpoint path (inside learner_group/learner/rl_module/default_policy)
    checkpoint_path = Path(checkpoint_path)
    module_path = checkpoint_path / "learner_group" / "learner" / "rl_module" / "default_policy"

    if not module_path.exists():
        print(f"Could not find RLModule at: {module_path}")
        return

    # Load the RLModule directly (no Ray needed for inference)
    print(f"Loading RLModule from: {module_path}")
    module = RLModule.from_checkpoint(str(module_path))
    module.eval()

    # Create evaluation environment
    env = create_env({})

    print(f"\n=== Evaluating DreamerV3 Agent ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Games to play: {num_games}")

    wins = 0
    total_rewards = []
    episode_lengths = []

    for game in range(num_games):
        obs, info = env.reset(seed=game)
        episode_reward = 0.0
        episode_length = 0
        done = False
        won = False

        # Initialize DreamerV3 hidden state
        state = module.get_initial_state()
        state = {k: v.unsqueeze(0) for k, v in state.items()}  # Add batch dim
        is_first = True

        while not done:
            # Convert observation to tensor with batch and time dimensions
            # DreamerV3 expects [B, T, obs_dim] for inference
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            is_first_tensor = torch.tensor([1.0 if is_first else 0.0])

            # Get action from the trained policy
            with torch.no_grad():
                actions, state = module.dreamer_model.forward_inference(
                    observations=obs_tensor,
                    previous_states=state,
                    is_first=is_first_tensor,
                )

            # Actions are one-hot encoded, convert to discrete action
            action = actions.squeeze(0).argmax().item()
            is_first = False

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            if done:
                won = info.get("won", False)
                if won:
                    wins += 1

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        status = "WIN" if won else "LOSE"
        print(f"Game {game + 1}/{num_games}: {status}, reward={episode_reward:.2f}, length={episode_length}")

    # Save replay of last game
    replay_data = env._env.game.clients[0].save_replay()
    checkpoint_name = checkpoint_path.parent.name
    replay_dir = Path("artifacts/replays/dreamerv3") / checkpoint_name
    replay_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    replay_path = replay_dir / f"{timestamp}.SC2Replay"
    replay_path.write_bytes(replay_data)
    print(f"\nReplay saved to: {replay_path}")

    env.close()

    # Print summary
    win_rate = wins / num_games * 100
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)

    print(f"\n=== Evaluation Results ({num_games} games) ===")
    print(f"Win rate: {wins}/{num_games} ({win_rate:.1f}%)")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average episode length: {avg_length:.1f}")
    print(f"Best reward: {max(total_rewards):.2f}")
    print(f"Worst reward: {min(total_rewards):.2f}")


def demo():
    """Quick demo to test the environment works."""
    print("=== Demo: Testing SC2GymEnv ===")

    # Create environment
    env = create_env({})

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial obs shape: {obs.shape}")
    print(f"Initial obs:\n{obs}")

    # Run a few steps with random actions
    for step in range(10):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1}: reward={reward:.3f}, term={terminated}, trunc={truncated}")

        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()

    env.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamerV3 training for SC2")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train DreamerV3 agent")
    train_parser.add_argument(
        "--num-envs", type=int, default=2, help="Number of parallel SC2 game instances"
    )
    train_parser.add_argument(
        "--steps", type=int, default=100_000, help="Total environment steps"
    )
    train_parser.add_argument(
        "--model-size",
        type=str,
        default="XS",
        choices=["XS", "S", "M", "L", "XL"],
        help="DreamerV3 model size",
    )
    train_parser.add_argument(
        "--training-ratio", type=int, default=32, help="Training ratio (32=fast, 512=paper)"
    )
    train_parser.add_argument(
        "--checkpoint-freq", type=int, default=10, help="Checkpoint frequency"
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint (uses latest if not specified)"
    )
    eval_parser.add_argument(
        "--games", type=int, default=10, help="Number of games to play"
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run quick demo")

    args = parser.parse_args()

    if args.command == "train":
        train(
            num_envs=args.num_envs,
            steps=args.steps,
            model_size=args.model_size,
            training_ratio=args.training_ratio,
            checkpoint_freq=args.checkpoint_freq,
            seed=args.seed,
        )
    elif args.command == "eval":
        eval(
            checkpoint_path=args.checkpoint,
            num_games=args.games,
        )
    elif args.command == "demo":
        demo()
    else:
        parser.print_help()
