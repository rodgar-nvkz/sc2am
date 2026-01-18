"""Evaluation script for entity-attention based model with hybrid action space."""

import glob
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from scam.settings import PROJECT_ROOT

from .env import SC2GymEnv
from .model import ActorCritic, ModelConfig


def eval_model(num_games: int = 10, model_path: str | None = None, use_cuda: bool = False):
    """Evaluate a trained model with hybrid action space using deterministic policy."""
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_path is None:
        # Try to find the latest model
        patterns = ["artifacts/models/smarte_v1_*.pt", "artifacts/models/lstm_v0_*.pt"]
        model_files: list[str] = []
        for pattern in patterns:
            model_files.extend(glob.glob(pattern))
        if not model_files:
            raise FileNotFoundError("No model checkpoint found in artifacts/models/")
        model_path = max(model_files, key=os.path.getctime)

    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get("model_config", {})
    model_config = ModelConfig(**config_dict) if config_dict else ModelConfig()
    model = ActorCritic(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wins = 0
    total_rewards = []
    total_lengths = []
    env = SC2GymEnv()

    logger.info(f"\nEvaluating for {num_games} games...")
    for game in range(num_games):
        done = False
        episode_length = 0
        episode_reward = 0.0
        obs, info = env.reset()

        # Initialize GRU hidden state for new episode
        hidden = model.get_initial_hidden(batch_size=1, device=device)

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

                # Get action mask from env and convert to model format
                env_action_mask = info.get("action_mask", None)
                action_mask_tensor = None
                range_mask = None

                if env_action_mask is not None:
                    # Convert env mask [MOVE, ATTACK_Z1, ATTACK_Z2, STOP] to model mask [MOVE, ATTACK, STOP]
                    model_mask = np.array([
                        env_action_mask[0],  # MOVE
                        env_action_mask[1] or env_action_mask[2],  # ATTACK
                        env_action_mask[3] if len(env_action_mask) > 3 else True,  # STOP
                    ], dtype=bool)
                    action_mask_tensor = torch.from_numpy(model_mask).unsqueeze(0).to(device)
                    range_mask = torch.tensor([[env_action_mask[1], env_action_mask[2]]], dtype=torch.bool, device=device)

                # Use deterministic action for evaluation
                action, hidden = model.get_deterministic_action(
                    obs_tensor,
                    hidden=hidden,
                    action_mask=action_mask_tensor,
                    range_mask=range_mask,
                )

            # Convert to environment action format
            env_action = model.to_env_action(action, batch_idx=0)

            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            episode_length += 1
            episode_reward += reward

        wins += info["won"]
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        print(f"Game {game + 1}: reward={episode_reward:.2f}, length={episode_length}, won={info['won']}")

    # Save replay of last game
    replay_data = env.game.clients[0].save_replay()
    assert model_path is not None
    replay_dir = PROJECT_ROOT / "artifacts" / "replays" / Path(model_path).stem
    replay_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    replay_path = replay_dir / f"{timestamp}.SC2Replay"
    replay_path.write_bytes(replay_data)
    print(f"Replay saved to {replay_path}")
    env.close()

    print(f"\n=== Results over {num_games} games ===")
    print(f"Win rate: {wins}/{num_games} ({100 * wins / num_games:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average length: {np.mean(total_lengths):.1f} ± {np.std(total_lengths):.1f}")


def eval_model_stochastic(num_games: int = 10, model_path: str | None = None, use_cuda: bool = False):
    """Evaluate using stochastic policy (sampling from distribution)."""
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_path is None:
        patterns = ["artifacts/models/smarte_v1_*.pt", "artifacts/models/lstm_v0_*.pt"]
        model_files: list[str] = []
        for pattern in patterns:
            model_files.extend(glob.glob(pattern))
        if not model_files:
            raise FileNotFoundError("No model checkpoint found in artifacts/models/")
        model_path = max(model_files, key=os.path.getctime)

    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get("model_config", {})
    model_config = ModelConfig(**config_dict) if config_dict else ModelConfig()
    model = ActorCritic(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wins = 0
    total_rewards = []
    total_lengths = []
    env = SC2GymEnv()

    logger.info(f"\nEvaluating (stochastic) for {num_games} games...")
    for game in range(num_games):
        done = False
        episode_length = 0
        episode_reward = 0.0
        obs, info = env.reset()

        # Initialize GRU hidden state for new episode
        hidden = model.get_initial_hidden(batch_size=1, device=device)

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

                # Get action mask from env and convert to model format
                env_action_mask = info.get("action_mask", None)
                action_mask_tensor = None
                range_mask = None

                if env_action_mask is not None:
                    # Convert env mask [MOVE, ATTACK_Z1, ATTACK_Z2, STOP] to model mask [MOVE, ATTACK, STOP]
                    model_mask = np.array([
                        env_action_mask[0],  # MOVE
                        env_action_mask[1] or env_action_mask[2],  # ATTACK
                        env_action_mask[3] if len(env_action_mask) > 3 else True,  # STOP
                    ], dtype=bool)
                    action_mask_tensor = torch.from_numpy(model_mask).unsqueeze(0).to(device)
                    range_mask = torch.tensor([[env_action_mask[1], env_action_mask[2]]], dtype=torch.bool, device=device)

                # Use stochastic action (sample from distribution)
                output = model(
                    obs_tensor,
                    hidden=hidden,
                    action_mask=action_mask_tensor,
                    range_mask=range_mask,
                )
                hidden = output.hidden

            # Convert to environment action format
            env_action = model.to_env_action(output.action, batch_idx=0)

            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            episode_length += 1
            episode_reward += reward

        wins += info["won"]
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        print(f"Game {game + 1}: reward={episode_reward:.2f}, length={episode_length}, won={info['won']}")

    env.close()

    print(f"\n=== Results over {num_games} games (stochastic) ===")
    print(f"Win rate: {wins}/{num_games} ({100 * wins / num_games:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average length: {np.mean(total_lengths):.1f} ± {np.std(total_lengths):.1f}")
