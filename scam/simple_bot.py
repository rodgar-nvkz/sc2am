"""A simple StarCraft II bot that builds workers and expands."""

import argparse
import asyncio

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer

from scam.remote_client import run_game_remote, run_game_remote_sync


class SimpleBot(BotAI):
    async def on_step(self, iteration: int):
        # After 50 iterations (~3.5 seconds), send some workers to attack
        if iteration == 50:
            for worker in self.workers.random_group_of(5):
                if self.enemy_start_locations:
                    worker.attack(self.enemy_start_locations[0])


def main():
    """Run the bot either locally or on a remote server."""
    parser = argparse.ArgumentParser(description="Simple SC2 Bot")
    parser.add_argument(
        "--remote", type=str, default="ws://192.168.42.122:40123/sc2api"
    )
    parser.add_argument(
        "--map", type=str, default="Ladder2019Season1/AutomatonLE.SC2Map"
    )
    args = parser.parse_args()
    print(f"Connecting to remote server: {args.remote}")
    print(f"Using map: {args.map}")

    remoute_game = run_game_remote(
        url=args.remote,
        map_path=args.map,
        bot=Bot(Race.Terran, SimpleBot()),
        opponent=Computer(Race.Protoss, Difficulty.Easy),
    )
    game_result = asyncio.run(remoute_game)
    print(f"Game result: {game_result}")


if __name__ == "__main__":
    main()
