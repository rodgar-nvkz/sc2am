"""Run bot vs bot matches in StarCraft II."""

from sc2 import maps
from sc2.data import Race
from sc2.main import run_game
from sc2.player import Bot

from scam.simple_bot import SimpleBot


def main():
    run_game(
        maps.get("(2)CatalystLE"),
        [
            Bot(Race.Terran, SimpleBot()),
            Bot(Race.Protoss, SimpleBot()),
        ],
        realtime=False,
        save_replay_as="bot_vs_bot_replay.SC2Replay",
    )

    print("\nMatch complete! Replay saved as: bot_vs_bot_replay.SC2Replay")


if __name__ == "__main__":
    main()
