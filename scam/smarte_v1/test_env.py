"""
Test script to verify environment actions work correctly.

Run with: python -m scam.smarte_v0.test_env
"""

import math
import time

import numpy as np

from .env import MARINE_RANGE, SC2GymEnv


def print_obs(obs: np.ndarray, step: int, action: int | None = None) -> None:
    """Pretty print observation."""
    action_names = ["MOVE_E", "MOVE_N", "MOVE_W", "MOVE_S", "ATK_Z1", "ATK_Z2", "STOP", "SKIP"]
    action_str = action_names[action] if action is not None else "N/A"

    time_remaining = obs[0]
    marine_hp = obs[1]
    weapon_cd = obs[2]
    weapon_cd_norm = obs[3]

    z1_hp = obs[4]
    z1_sin = obs[5]
    z1_cos = obs[6]
    z1_dist_norm = obs[7]
    z1_dist = z1_dist_norm * 30.0
    z1_angle = math.degrees(math.atan2(z1_sin, z1_cos))

    z2_hp = obs[8]
    z2_sin = obs[9]
    z2_cos = obs[10]
    z2_dist_norm = obs[11]
    z2_dist = z2_dist_norm * 30.0
    z2_angle = math.degrees(math.atan2(z2_sin, z2_cos))

    print(f"\n{'='*60}")
    print(f"Step {step:3d} | Action: {action_str:8s}")
    print(f"{'='*60}")
    print(f"Time remaining: {time_remaining:.2f}")
    print(f"Marine HP: {marine_hp:.2f} | Weapon CD: {weapon_cd:.0f} ({weapon_cd_norm:.2f})")
    print(f"Z1: HP={z1_hp:.2f}, Dist={z1_dist:.1f} (range={MARINE_RANGE}), Angle={z1_angle:.0f}°")
    print(f"Z2: HP={z2_hp:.2f}, Dist={z2_dist:.1f} (range={MARINE_RANGE}), Angle={z2_angle:.0f}°")


def test_all_actions():
    """Test each action type and verify it has an effect."""
    print("\n" + "="*60)
    print("TEST: Verify all actions have observable effects")
    print("="*60)

    env = SC2GymEnv({"num_move_directions": 4})

    try:
        # Test each action
        for action in range(env.num_actions):
            obs, _ = env.reset()
            print_obs(obs, 0, None)

            # Take the action
            obs2, reward, terminated, truncated, info = env.step(action)
            print_obs(obs2, 1, action)

            # Compare observations
            diff = np.abs(obs2 - obs)
            changed_indices = np.where(diff > 0.001)[0]

            action_names = ["MOVE_E", "MOVE_N", "MOVE_W", "MOVE_S", "ATK_Z1", "ATK_Z2", "STOP", "SKIP"]
            obs_names = ["time", "marine_hp", "wpn_cd", "wpn_cd_norm",
                        "z1_hp", "z1_sin", "z1_cos", "z1_dist",
                        "z2_hp", "z2_sin", "z2_cos", "z2_dist"]

            print(f"\nAction {action} ({action_names[action]}): Changed indices = {changed_indices}")
            for idx in changed_indices:
                print(f"  {obs_names[idx]}: {obs[idx]:.3f} -> {obs2[idx]:.3f} (Δ={diff[idx]:.3f})")

            if len(changed_indices) <= 1:  # Only time changed
                print(f"  ⚠️  WARNING: Action {action_names[action]} had no effect besides time!")

            print("-"*60)
            time.sleep(0.5)

    finally:
        env.close()


def test_movement_directions():
    """Test that movement actions move in correct directions."""
    print("\n" + "="*60)
    print("TEST: Verify movement directions are correct")
    print("="*60)

    env = SC2GymEnv({"num_move_directions": 4})

    try:
        # For each move direction, check that distance to zerglings changes appropriately
        for action in range(4):  # 0=E, 1=N, 2=W, 3=S
            obs, _ = env.reset()

            # Get initial zergling positions (via angle/distance)
            z1_dist_before = obs[7] * 30.0
            z1_angle_before = math.degrees(math.atan2(obs[5], obs[6]))

            # Take multiple steps in same direction
            for step in range(5):
                obs, _, terminated, _, _ = env.step(action)
                if terminated:
                    break

            z1_dist_after = obs[7] * 30.0
            z1_angle_after = math.degrees(math.atan2(obs[5], obs[6]))

            action_names = ["MOVE_E (0°)", "MOVE_N (90°)", "MOVE_W (180°)", "MOVE_S (270°)"]
            print(f"\n{action_names[action]}:")
            print(f"  Z1 distance: {z1_dist_before:.1f} -> {z1_dist_after:.1f}")
            print(f"  Z1 angle:    {z1_angle_before:.0f}° -> {z1_angle_after:.0f}°")

    finally:
        env.close()


def test_attack_cooldown():
    """Test that attacking triggers cooldown."""
    print("\n" + "="*60)
    print("TEST: Verify attack triggers weapon cooldown")
    print("="*60)

    env = SC2GymEnv({"num_move_directions": 4})

    try:
        obs, _ = env.reset()
        print_obs(obs, 0, None)

        # Attack Z1
        for step in range(10):
            obs, reward, terminated, truncated, info = env.step(4)  # ATK_Z1
            print_obs(obs, step + 1, 4)

            if terminated:
                print(f"\nEpisode ended! Won: {info.get('won', False)}")
                break

            time.sleep(0.2)

    finally:
        env.close()


def test_kite_sequence():
    """Test a manual kite sequence: attack -> move away -> attack."""
    print("\n" + "="*60)
    print("TEST: Manual kite sequence")
    print("="*60)

    env = SC2GymEnv({"num_move_directions": 4})

    try:
        obs, _ = env.reset()
        print_obs(obs, 0, None)

        # Kite pattern: find zergling direction, attack, move opposite
        step = 0
        while True:
            # Get zergling angle to decide kite direction
            z1_sin, z1_cos = obs[5], obs[6]
            z1_angle = math.atan2(z1_sin, z1_cos)
            z1_angle_deg = math.degrees(z1_angle)

            weapon_cd = obs[2]
            z1_dist = obs[7] * 30.0

            # Decide action
            if weapon_cd == 0 and z1_dist < MARINE_RANGE:
                # Weapon ready and in range - attack!
                action = 4  # ATK_Z1
            else:
                # Move away from zergling
                # Zergling is at angle z1_angle, so move opposite
                opposite_angle = (z1_angle + math.pi) % (2 * math.pi)
                # Convert to discrete direction (0=E, 1=N, 2=W, 3=S)
                # Each direction covers 90 degrees
                action = int((opposite_angle + math.pi/4) / (math.pi/2)) % 4

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            print_obs(obs, step, action)

            if terminated or truncated:
                print(f"\nEpisode ended at step {step}! Won: {info.get('won', False)}, Reward: {reward}")
                break

            if step > 100:
                print("\nStopping after 100 steps")
                break

            time.sleep(0.1)

    finally:
        env.close()


def test_random_policy():
    """Test random policy to see distribution of outcomes."""
    print("\n" + "="*60)
    print("TEST: Random policy statistics (20 episodes)")
    print("="*60)

    env = SC2GymEnv({"num_move_directions": 4})

    try:
        wins = 0
        total_rewards = []
        episode_lengths = []
        action_counts = [0] * env.num_actions

        for ep in range(20):
            obs, _ = env.reset()
            ep_reward = 0
            steps = 0

            while True:
                action = env.action_space.sample()
                action_counts[action] += 1

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1

                if terminated or truncated:
                    if info.get('won', False):
                        wins += 1
                    total_rewards.append(ep_reward)
                    episode_lengths.append(steps)
                    break

            print(f"Episode {ep+1}: steps={steps}, reward={ep_reward:.2f}, won={info.get('won', False)}")

        print(f"\nSummary:")
        print(f"  Win rate: {wins}/{20} = {wins/20*100:.1f}%")
        print(f"  Avg reward: {np.mean(total_rewards):.2f}")
        print(f"  Avg length: {np.mean(episode_lengths):.1f}")
        print(f"  Action distribution: {action_counts}")

    finally:
        env.close()


def main():
    """Run all tests."""
    print("SC2 Environment Debug Tests")
    print("="*60)

    tests = [
        ("1. All Actions Effect", test_all_actions),
        ("2. Movement Directions", test_movement_directions),
        ("3. Attack Cooldown", test_attack_cooldown),
        ("4. Manual Kite", test_kite_sequence),
        ("5. Random Policy", test_random_policy),
    ]

    print("\nAvailable tests:")
    for i, (name, _) in enumerate(tests):
        print(f"  {name}")

    print("\nSelect test (1-5, or 'a' for all): ", end="")
    choice = input().strip().lower()

    if choice == 'a':
        for name, test_fn in tests:
            test_fn()
    elif choice.isdigit() and 1 <= int(choice) <= len(tests):
        tests[int(choice) - 1][1]()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
