"""Run bot vs bot matches in StarCraft II."""

import datetime
import os

from sc2 import maps
from sc2.data import Race
from sc2.main import run_game
from sc2.player import Bot

from scam.simple_bot import SimpleBot


def main():
    replay_path = f"replays/{datetime.datetime.now().isoformat()}.SC2Replay"
    run_game(
        maps.get("(2)CatalystLE"),
        [
            Bot(Race.Terran, SimpleBot()),
            Bot(Race.Protoss, SimpleBot()),
        ],
        realtime=False,
        game_time_limit=60 * 5,
        save_replay_as=replay_path,
    )


if __name__ == "__main__":
    main()
