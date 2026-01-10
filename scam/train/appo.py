import argparse
from ray import tune
from ray.tune import CLIReporter
from ray.rllib.algorithms.appo import APPOConfig


class APPOCLIReported(CLIReporter):
    def __init__(self, *a, **kw):
        kw["metric_columns"] = {
            "env_runners/episode_reward_mean": "reward",
            "env_runners/episode_len_mean": "ep_len",
            "num_env_steps_sampled_lifetime": "total_steps",
            "num_env_steps_sampled_per_second": "steps/s",
        }
        super().__init__(*a, **kw)



def train(num_envs: int, num_steps: int) -> None:
    config = APPOConfig()
    config = config.training(lr=0.01, grad_clip=30.0, train_batch_size_per_learner=50)
    config = config.learners(num_learners=1)
    config = config.env_runners(num_env_runners=num_envs)
    config = config.environment("CartPole-v1")
    # config.build().train()

    run_config=tune.RunConfig(stop={"num_env_steps_sampled_lifetime": num_steps}, progress_reporter=APPOCLIReported())
    tuner = tune.Tuner("APPO", param_space=config.to_dict(), run_config=run_config)
    tuner.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate Marine vs Zergling RL agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument("--steps", type=int, default=100_000, help="Total training steps")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--num-envs", type=int, default=10, help="Number of parallel environments")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    eval_parser.add_argument("--model", type=str, default=None, help="Path to model file")

    # Demo command
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
