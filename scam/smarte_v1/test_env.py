"""
Test script for smarte_v1 environment.

Run with: python -m scam.smarte_v1.test_env [test_number|a]
Examples:
    python -m scam.smarte_v1.test_env 1    # Run movement test
    python -m scam.smarte_v1.test_env a    # Run all tests
    python -m scam.smarte_v1.test_env      # Interactive mode
"""

import math
import sys

import numpy as np

from .env import (
    ACTION_ATTACK_Z1,
    ACTION_ATTACK_Z2,
    ACTION_MOVE,
    ACTION_STOP,
    NUM_COMMANDS,
    SC2GymEnv,
)


def make_action(command: int, angle_rad: float = 0.0) -> dict:
    """Helper to create action dict."""
    return {
        "command": command,
        "angle": np.array([math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float32),
    }


def print_obs(obs: np.ndarray, step: int, action: dict | None = None, mask: np.ndarray | None = None) -> None:
    """Pretty print observation."""
    cmd_names = ["MOVE", "ATK_Z1", "ATK_Z2", "STOP"]

    if action is not None:
        cmd = action["command"]
        angle = math.degrees(math.atan2(action["angle"][0], action["angle"][1]))
        action_str = f"{cmd_names[cmd]} ({angle:.0f}°)"
    else:
        action_str = "N/A"

    time_left = obs[0]
    marine_hp = obs[1]
    weapon_cd = obs[2]
    weapon_cd_norm = obs[3]

    z1_hp, z1_sin, z1_cos, z1_dist_norm = obs[4:8]
    z2_hp, z2_sin, z2_cos, z2_dist_norm = obs[8:12]

    z1_dist = z1_dist_norm * 30.0
    z2_dist = z2_dist_norm * 30.0
    z1_angle = math.degrees(math.atan2(z1_sin, z1_cos))
    z2_angle = math.degrees(math.atan2(z2_sin, z2_cos))

    print(f"\n{'='*60}")
    print(f"Step {step:3d} | Action: {action_str}")
    if mask is not None:
        mask_str = " ".join(f"{cmd_names[i]}={'Y' if mask[i] else 'N'}" for i in range(NUM_COMMANDS))
        print(f"Mask: [{mask_str}]")
    print(f"{'='*60}")
    print(f"Time left: {time_left:.2f} | Marine HP: {marine_hp:.2f} | CD: {weapon_cd:.0f} ({weapon_cd_norm:.2f})")
    print(f"Z1: HP={z1_hp:.2f}, Dist={z1_dist:.1f}, Angle={z1_angle:.0f}°")
    print(f"Z2: HP={z2_hp:.2f}, Dist={z2_dist:.1f}, Angle={z2_angle:.0f}°")


def test_movement():
    """Test movement with sin/cos angles."""
    print("\n" + "=" * 60)
    print("TEST: Movement with sin/cos angles")
    print("=" * 60)

    env = SC2GymEnv()

    try:
        obs, info = env.reset()
        print_obs(obs, 0, None, info["action_mask"])

        # Move in 4 cardinal directions
        directions = [
            (0, "East (0°)"),
            (math.pi / 2, "North (90°)"),
            (math.pi, "West (180°)"),
            (-math.pi / 2, "South (-90°)"),
        ]

        done = False
        for angle, name in directions:
            print(f"\n--- Moving {name} for 3 steps ---")
            for _ in range(3):
                action = make_action(ACTION_MOVE, angle)
                obs, reward, terminated, truncated, info = env.step(action)
                print_obs(obs, env.current_step, action, info["action_mask"])
                if terminated:
                    done = True
                    break
            if done:
                break

    finally:
        env.close()


def test_attack():
    """Test attack actions and action masks."""
    print("\n" + "=" * 60)
    print("TEST: Attack and action masks")
    print("=" * 60)

    env = SC2GymEnv()

    try:
        obs, info = env.reset()
        print_obs(obs, 0, None, info["action_mask"])

        # Get angle to Z1 from observation
        z1_sin, z1_cos = obs[5], obs[6]
        angle_to_z1 = math.atan2(z1_sin, z1_cos)

        print(f"\n--- Moving toward Z1 (angle={math.degrees(angle_to_z1):.0f}°) ---")

        # Move toward zerglings until attack is available
        for _ in range(50):
            mask = info["action_mask"]

            if mask[ACTION_ATTACK_Z1]:
                print("\n--- Attack available! Attacking Z1 ---")
                action = make_action(ACTION_ATTACK_Z1)
            else:
                # Update angle from current obs
                z1_sin, z1_cos = obs[5], obs[6]
                angle_to_z1 = math.atan2(z1_sin, z1_cos)
                action = make_action(ACTION_MOVE, angle_to_z1)

            obs, reward, terminated, truncated, info = env.step(action)
            print_obs(obs, env.current_step, action, info["action_mask"])

            if terminated:
                print(f"\nEpisode ended! Reward: {reward:.3f}, Won: {info['won']}")
                break

    finally:
        env.close()


def test_kite():
    """Test manual kite sequence: move toward -> attack -> move away -> repeat."""
    print("\n" + "=" * 60)
    print("TEST: Manual kite sequence")
    print("=" * 60)

    env = SC2GymEnv()

    try:
        obs, info = env.reset()
        print_obs(obs, 0, None, info["action_mask"])

        for step in range(200):
            mask = info["action_mask"]
            weapon_cd = obs[2]

            # Get angle to closest zergling
            z1_dist = obs[7] * 30.0
            z2_dist = obs[11] * 30.0

            if z1_dist <= z2_dist and obs[4] > 0:  # Z1 closer and alive
                z_sin, z_cos = obs[5], obs[6]
            elif obs[8] > 0:  # Z2 alive
                z_sin, z_cos = obs[9], obs[10]
            else:  # Only Z1 left
                z_sin, z_cos = obs[5], obs[6]

            angle_to_enemy = math.atan2(z_sin, z_cos)
            angle_away = angle_to_enemy + math.pi  # Opposite direction

            # Kite logic
            if weapon_cd == 0 and (mask[ACTION_ATTACK_Z1] or mask[ACTION_ATTACK_Z2]):
                # Attack closest available target
                if mask[ACTION_ATTACK_Z1] and (not mask[ACTION_ATTACK_Z2] or z1_dist <= z2_dist):
                    action = make_action(ACTION_ATTACK_Z1)
                else:
                    action = make_action(ACTION_ATTACK_Z2)
            elif weapon_cd > 0:
                # Kite away while on cooldown
                action = make_action(ACTION_MOVE, angle_away)
            else:
                # Move toward enemy
                action = make_action(ACTION_MOVE, angle_to_enemy)

            obs, reward, terminated, truncated, info = env.step(action)

            if step % 10 == 0 or terminated:
                print_obs(obs, env.current_step, action, info["action_mask"])

            if terminated or truncated:
                print(f"\nEpisode ended! Reward: {reward:.3f}, Won: {info['won']}")
                break

    finally:
        env.close()


def test_random_policy():
    """Test random policy statistics."""
    print("\n" + "=" * 60)
    print("TEST: Random policy (10 episodes)")
    print("=" * 60)

    env = SC2GymEnv()

    try:
        wins = 0
        rewards = []
        lengths = []

        for ep in range(10):
            obs, info = env.reset()
            ep_reward = 0
            steps = 0

            while True:
                # Random action respecting mask
                mask = info["action_mask"]
                valid_commands = np.where(mask)[0]
                command = np.random.choice(valid_commands)
                angle = np.random.uniform(-math.pi, math.pi)
                action = make_action(command, angle)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1

                if terminated or truncated:
                    if info["won"]:
                        wins += 1
                    rewards.append(ep_reward)
                    lengths.append(steps)
                    print(f"Episode {ep + 1}: steps={steps}, reward={ep_reward:.3f}, won={info['won']}")
                    break

        print("\nSummary:")
        print(f"  Win rate: {wins}/10 = {wins * 10:.0f}%")
        print(f"  Avg reward: {np.mean(rewards):.3f}")
        print(f"  Avg length: {np.mean(lengths):.1f}")

    finally:
        env.close()


def main():
    tests = [
        ("1. Movement", test_movement),
        ("2. Attack", test_attack),
        ("3. Kite", test_kite),
        ("4. Random", test_random_policy),
    ]

    print("smarte_v1 Environment Tests")
    print("=" * 60)
    print("\nAvailable tests:")
    for name, _ in tests:
        print(f"  {name}")

    # Get choice from CLI args or interactive input
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("\nSelect test (1-4, or 'a' for all): ", end="")
        choice = input().strip().lower()

    if choice == "a":
        for _, fn in tests:
            fn()
    elif choice.isdigit() and 1 <= int(choice) <= len(tests):
        tests[int(choice) - 1][1]()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
