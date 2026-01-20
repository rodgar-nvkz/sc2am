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
from .obs import ObsSpec


def eval_model(num_games: int = 10, model_path: str | None = None, upgrade_level: int = 1):
    """Evaluate a trained model with hybrid action space."""
    device = torch.device("cpu")

    if model_path is None:
        model_path = max(glob.glob("artifacts/models/smarte_v3_*.pt"), key=os.path.getctime)

    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Reconstruct ModelConfig with ObsSpec
    saved_config = checkpoint["model_config"]
    saved_config["obs_spec"] = ObsSpec(**saved_config["obs_spec"])

    model_config = ModelConfig(**saved_config)
    model = ActorCritic(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wins = 0
    total_rewards = []
    total_lengths = []
    env = SC2GymEnv({"upgrade_level": [upgrade_level], "game_steps_per_env": 2})

    logger.info(f"\nEvaluating for {num_games} games...")
    for game in range(num_games):
        done = False
        episode_length = 0
        episode_reward = 0.0
        obs, info = env.reset()
        action_mask = info["action_mask"]

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                mask_tensor = torch.from_numpy(action_mask).unsqueeze(0)
                # Use deterministic action for evaluation (with action masking)
                command, angle = model.get_deterministic_action(obs_tensor, action_mask=mask_tensor)

            action = {"command": command.squeeze(0).item(), "angle": angle.squeeze(0).numpy()}
            obs, reward, terminated, truncated, info = env.step(action)
            action_mask = info["action_mask"]
            done = terminated or truncated
            episode_length += 1
            episode_reward += reward

        wins += episode_reward > 0
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        print(f"Game {game + 1}: reward={episode_reward:.2f}, length={episode_length}, won={episode_reward > 0}")

    # Save replay of last game
    replay_data = env.game.clients[0].save_replay()
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
