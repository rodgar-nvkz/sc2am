import atexit
import time
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from s2clientprotocol.common_pb2 import Race

from scam.infra.client import PortConfig, SC2Client
from scam.infra.server import SC2LocalhostServer
from scam.maps.generator import MapGenerator


class SC2MultiplayerGame:
    RACES: list[int] = [Race.Terran, Race.Zerg, Race.Protoss]

    def __init__(self, races: list[int]) -> None:
        self.maps = MapGenerator()
        self.races = races
        self.servers: list[SC2LocalhostServer] = []
        self.clients: list[SC2Client] = []
        self.pool = ThreadPoolExecutor(max_workers=len(races))
        self.port_config = PortConfig.pick_ports(len(self.races))
        atexit.register(self.close)

    def join_args(self, i) -> tuple:
        return self.races[i], self.port_config, None if i == 0 else self.servers[0].host

    def launch(self) -> "SC2MultiplayerGame":
        for _ in range(len(self.races)):
            server = SC2LocalhostServer()
            self.servers.append(server)
            self.clients.append(SC2Client(server.socket))

        self.clients[0].host_game(next(self.maps), players=self.races)
        list(self.pool.map(lambda i: self.clients[i].join_game(*self.join_args(i)), range(len(self.races))))
        logger.info("All players have joined the game")
        return self

    def step(self) -> int:
        steps = list(self.pool.map(lambda c: c.step().simulation_loop, self.clients))
        assert len(set(steps)) == 1, "All clients must return the same step count"
        return steps[0]

    def perf(self, seconds: int = 5) -> None:
        steps, start = 0, time.time()
        while time.time() - start < seconds:
            self.step()
            steps += 1
        duration = time.time() - start
        logger.info(f"Performance: {steps / duration:.2f} steps/sec over {duration:.2f} seconds")

    def close(self) -> None:
        for server in self.servers:
            server.close()
