import argparse

from ray import tune
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune import Callback, CLIReporter
from ray.tune.registry import register_env

from scam.envs.gym import SC2GymEnv

register_env("sc2_mvz_v1", lambda cfg: SC2GymEnv())



class MetricPrinter(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        env_runners = result.get("env_runners", {})

        reward = env_runners.get("episode_return_mean")
        steps = env_runners.get("num_env_steps_sampled_lifetime")
        throughput = env_runners.get("num_env_steps_sampled_lifetime_throughput", {})
        steps_per_sec = throughput.get("throughput_since_last_reduce")

        reward_str = f"{reward:7.2f}" if isinstance(reward, (int, float)) else "    N/A"
        steps_str = f"{int(steps):,}" if isinstance(steps, (int, float)) else "N/A"
        steps_per_sec_str = f"{steps_per_sec:.0f}" if isinstance(steps_per_sec, (int, float)) else "N/A"

        print(f"Iter {result['training_iteration']:3d} | Reward: {reward_str} | Steps: {steps_str} | Steps/s: {steps_per_sec_str}")



def train(num_envs: int, num_steps: int) -> None:
    config = APPOConfig()
    config = config.training(lr=0.01, grad_clip=30.0, train_batch_size_per_learner=50)
    config = config.learners(num_learners=1, num_gpus_per_learner=1)
    config = config.env_runners(num_env_runners=num_envs)
    config = config.environment("sc2_mvz_v1")
    # config.build_algo().train()

    run_config=tune.RunConfig(stop={"num_env_steps_sampled_lifetime": num_steps}, callbacks=[MetricPrinter()], verbose=0)
    tuner = tune.Tuner("APPO", param_space=config.to_dict(), run_config=run_config)
    tuner.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate Marine vs Zergling RL agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument("--steps", type=int, default=100_000, help="Total training steps")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--num-envs", type=int, default=10, help="Number of parallel environments")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    eval_parser.add_argument("--model", type=str, default=None, help="Path to model file")

    demo_parser = subparsers.add_parser("demo", help="Run a quick demo")

    args = parser.parse_args()
    if args.command == "train":
        train(num_envs=args.num_envs, num_steps=args.steps)
    elif args.command == "eval":
        pass
        # eval(num_games=args.games, model_path=args.model)
    elif args.command == "demo":
        pass
        # demo()
