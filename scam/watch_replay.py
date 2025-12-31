"""Watch SC2 replay files."""

import argparse
import asyncio
from pathlib import Path

from sc2.bot_ai import BotAI
from sc2.main import _host_replay, get_replay_version


class ReplayObserver(BotAI):
    """Observer bot for watching replays."""

    async def on_step(self, iteration: int):
        """Called every game step during replay."""
        if iteration % 100 == 0:  # Print every 100 iterations to avoid spam
            print(
                f"Loop: {self.state.game_loop:>6} | "
                f"Time: {self.time_formatted:>8} | "
                f"Units: {len(self.units):>3} | "
                f"Workers: {len(self.workers):>2} | "
                f"Minerals: {self.minerals:>5}"
            )


def main():
    parser = argparse.ArgumentParser(description="Watch SC2 Replay")
    parser.add_argument(
        "replay_path",
        type=str,
        help="Absolute path to the replay file",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Watch at normal game speed (default: fast forward)",
    )
    parser.add_argument(
        "--player",
        type=int,
        default=0,
        help="Player ID to observe (0 or 1, default: 0)",
    )
    args = parser.parse_args()

    replay_path = Path(args.replay_path).resolve()
    assert replay_path.is_file(), f"Replay file not found: {replay_path}"
    assert replay_path.is_absolute(), "Replay path must be absolute"

    print(f"Loading replay: {replay_path}")
    print(f"Observing player: {args.player}")
    print(f"Realtime: {args.realtime}")
    print()

    base_build, data_version = get_replay_version(str(replay_path))

    asyncio.run(
        _host_replay(
            str(replay_path),
            ReplayObserver(),
            args.realtime,
            None,  # portconfig
            base_build,
            data_version,
            args.player,
        )
    )


if __name__ == "__main__":
    main()
