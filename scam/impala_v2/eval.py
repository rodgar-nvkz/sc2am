import glob
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from scam.envs.impala_v2 import NUM_COMMANDS, OBS_SIZE, SC2GymEnv
from scam.impala_v2.model import ActorCritic
from scam.settings import PROJECT_ROOT


def eval_model(num_games: int = 10, model_path: str | None = None, upgrade_level: int = 0):
    """Evaluate a trained model with hybrid action space."""
    device = torch.device("cpu")

    if model_path is None:
        model_path = max(glob.glob("artifacts/models/impala_v2_*.pt"), key=os.path.getctime)

    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model = ActorCritic(OBS_SIZE, NUM_COMMANDS).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wins = 0
    total_rewards = []
    total_lengths = []
    env = SC2GymEnv({"upgrade_level": [upgrade_level]})

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
                command, angle = model.get_deterministic_action(obs_tensor, mask_tensor)

            action = {'command': command.squeeze(0).item(), 'angle': angle.squeeze(0).numpy()}
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
