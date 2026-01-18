"""Evaluation script for LSTM-based model with discrete actions."""

import glob
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from smarte.settings import PROJECT_ROOT

from .env import SC2GymEnv
from .model import ActorCritic, ModelConfig


def eval_model(num_games: int = 10, model_path: str | None = None, upgrade_level: int = 0, use_cuda: bool = False):
    """Evaluate a trained LSTM model with discrete action space."""
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_path is None:
        # Try to find the latest model
        patterns = ["artifacts/models/lstm_v0_*.pt", "artifacts/models/impala_v2_*.pt"]
        model_files: list[str] = []
        for pattern in patterns:
            model_files.extend(glob.glob(pattern))
        if not model_files:
            raise FileNotFoundError("No model checkpoint found in artifacts/models/")
        model_path = max(model_files, key=os.path.getctime)

    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Filter out computed fields that are now generated in __post_init__
    config_dict = checkpoint["model_config"]
    computed_fields = {"num_actions", "action_attack_z1", "action_attack_z2", "action_stop", "action_skip"}
    filtered_config = {k: v for k, v in config_dict.items() if k not in computed_fields}
    model_config = ModelConfig(**filtered_config)
    model = ActorCritic(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wins = 0
    total_rewards = []
    total_lengths = []
    env = SC2GymEnv({
        "upgrade_level": [upgrade_level],
        "num_move_directions": model_config.num_move_directions,
    })

    logger.info(f"\nEvaluating for {num_games} games...")
    for game in range(num_games):
        done = False
        episode_length = 0
        episode_reward = 0.0
        obs, info = env.reset()

        # Initialize LSTM hidden state for new episode
        hidden = model.get_initial_hidden(batch_size=1, device=device)

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # Use deterministic action for evaluation
                action, hidden = model.get_deterministic_action(obs_tensor, hidden=hidden)

            action_int: int = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated
            episode_length += 1
            episode_reward += reward

        wins += info["won"]
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        print(f"Game {game + 1}: reward={episode_reward:.2f}, length={episode_length}, won={episode_reward > 0}")

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


def eval_model_stochastic(num_games: int = 10, model_path: str | None = None, upgrade_level: int = 0, use_cuda: bool = False):
    """Evaluate using stochastic policy (sampling from distribution)."""
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_path is None:
        patterns = ["artifacts/models/lstm_v0_*.pt", "artifacts/models/impala_v2_*.pt"]
        model_files: list[str] = []
        for pattern in patterns:
            model_files.extend(glob.glob(pattern))
        if not model_files:
            raise FileNotFoundError("No model checkpoint found in artifacts/models/")
        model_path = max(model_files, key=os.path.getctime)

    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Filter out computed fields that are now generated in __post_init__
    config_dict = checkpoint["model_config"]
    computed_fields = {"num_actions", "action_attack_z1", "action_attack_z2", "action_stop", "action_skip"}
    filtered_config = {k: v for k, v in config_dict.items() if k not in computed_fields}
    model_config = ModelConfig(**filtered_config)
    model = ActorCritic(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wins = 0
    total_rewards = []
    total_lengths = []
    env = SC2GymEnv({
        "upgrade_level": [upgrade_level],
        "num_move_directions": model_config.num_move_directions,
    })

    logger.info(f"\nEvaluating (stochastic) for {num_games} games...")
    for game in range(num_games):
        done = False
        episode_length = 0
        episode_reward = 0.0
        obs, info = env.reset()

        # Initialize LSTM hidden state for new episode
        hidden = model.get_initial_hidden(batch_size=1, device=device)

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # Use stochastic action (sample from distribution)
                output = model(obs_tensor, hidden=hidden)
                hidden = output.hidden

            action_int: int = int(output.action.action.item())
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated
            episode_length += 1
            episode_reward += reward

        wins += episode_reward > 0
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        print(f"Game {game + 1}: reward={episode_reward:.2f}, length={episode_length}, won={episode_reward > 0}")

    env.close()

    print(f"\n=== Results over {num_games} games (stochastic) ===")
    print(f"Win rate: {wins}/{num_games} ({100 * wins / num_games:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average length: {np.mean(total_lengths):.1f} ± {np.std(total_lengths):.1f}")
