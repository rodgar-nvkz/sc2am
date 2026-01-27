import atexit
import time
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from s2clientprotocol.common_pb2 import Protoss, Race, Terran, Zerg

from smarte.infra.client import PortConfig, SC2Client
from smarte.infra.server import GAME_FOLDER, SC2LocalhostServer
from smarte.maps.generator import MapGenerator


class SC2SingleGame:
    RACES: list[Race] = [Terran, Zerg, Protoss]
    SAFE_GAME_STEPS = 22.4 * 3600 * 6  # Prevents SC2 API Error: ['Game has already ended']

    def __init__(self, races: list[Race]) -> None:
        self.maps = MapGenerator()
        self.races = races
        self.servers: list[SC2LocalhostServer] = []
        self.clients: list[SC2Client] = []
        self.pool = ThreadPoolExecutor(max_workers=len(races))
        self.port_config = PortConfig.pick_ports(len(self.races))
        self.game_step: int = 0
        atexit.register(self.close)

    def launch(self) -> "SC2SingleGame":
        map_name, map_data = next(self.maps)
        with open(GAME_FOLDER / "maps" / "smarte" / map_name, "wb") as map:
            map.write(map_data)

        self.servers.append(SC2LocalhostServer())
        self.clients.append(SC2Client(self.servers[0].socket))
        players = [SC2Client.player(self.races[0], True)]
        players += [SC2Client.player(r, False) for r in self.races[1:]]
        self.clients[0].host_game(f"smarte/{map_name}", players=players)
        self.clients[0].join_game(self.races[0], None, None)
        logger.debug("Single player have joined the game")
        return self

    def step(self, count: int = 1) -> None:
        self.game_step += count
        self.clients[0].step(count)

    def reset_map(self) -> bool:
        """Reset the current game if it is required"""
        if self.game_step >= self.SAFE_GAME_STEPS:
            logger.debug(f"Resetting map after {self.game_step} steps (limit {self.SAFE_GAME_STEPS})")
            self.clients[0].restart_game()
            self.game_step = 0
            return True
        return False

    def perf(self, seconds: int = 5, obs: bool = False) -> None:
        steps, start = 0, time.time()
        while time.time() - start < seconds:
            self.step()
            self.clients[0].get_observation() if obs else None
            steps += 1
        duration = time.time() - start
        logger.info(f"Performance: {steps / duration:.2f} steps/sec over {duration:.2f} seconds")

    def close(self) -> None:
        for server in self.servers:
            server.close()


class SC2MultiplayerGame(SC2SingleGame):
    def join_args(self, i) -> tuple:
        return self.races[i], self.port_config, None if i == 0 else self.servers[0].host

    def launch(self) -> "SC2MultiplayerGame":
        map_name, map_data = next(self.maps)
        with open(GAME_FOLDER / "maps" / "smarte" / map_name, "wb") as map:
            map.write(map_data)

        for _ in range(len(self.races)):
            server = SC2LocalhostServer()
            self.servers.append(server)
            self.clients.append(SC2Client(server.socket))

        players = [SC2Client.player(r, True) for r in self.races]
        self.clients[0].host_game(map_name, players=players)
        list(
            self.pool.map(
                lambda i: self.clients[i].join_game(*self.join_args(i)),
                range(len(self.races)),
            )
        )
        logger.info("All players have joined the game")
        return self

    def step(self, count: int = 1):
        list(self.pool.map(lambda c: c.step(count), self.clients))
