import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import portpicker
import websocket
from s2clientprotocol import sc2api_pb2 as pb
from s2clientprotocol.common_pb2 import Point2D, Race
from s2clientprotocol.debug_pb2 import (
    DebugCommand,
    DebugCreateUnit,
    DebugKillUnit,
)

from scam.maps.generator import MapGenerator

GAME_FOLDER = Path("~/StarCraftII/").expanduser()
GAME_ENTRYPOINT = Path("Versions/Base75689/SC2_x64")
SERVER_HOST = "127.0.0.1"


@dataclass
class SC2GameProp:
    map_path: str
    players: list[Race]


class SC2BackgroundServer:
    def __init__(self) -> None:
        self.players = [Race.Terran, Race.Zerg]
        self.maps = MapGenerator()
        self.port = portpicker.PickUnusedPort()
        self.server = self.run_server(SERVER_HOST, self.port)
        self.socket = websocket.create_connection(
            f"ws://{SERVER_HOST}:{self.port}/sc2api"
        )
        self.player_id: int = self.startup()

    def run_server(self, host: str, port: int):
        path = str(GAME_FOLDER / GAME_ENTRYPOINT)
        args = (path, "-listen", host, "-port", str(port), "-displayMode", "0")
        server = subprocess.Popen(args, encoding="utf8")
        time.sleep(3)
        return server

    def startup(self) -> int:
        """Returns the player ID after hosting and joining a game"""
        self.host_game()
        game_response = self.join_game()
        return game_response.player_id

    def host_game(self):
        players = [
            pb.PlayerSetup(
                race=Race.Terran,
                type=pb.PlayerType.Participant,
            ),
            pb.PlayerSetup(
                race=Race.Zerg,
                type=pb.PlayerType.Computer,
            ),
        ]
        local_map = pb.LocalMap(map_data=next(self.maps))
        request = pb.RequestCreateGame(
            local_map=local_map, player_setup=players, realtime=False
        )
        response = self.send(create_game=request).create_game
        assert not response.HasField("error"), response.error
        return response

    def step(self) -> pb.ResponseStep:
        return self.send(step=pb.RequestStep(count=1)).step

    def observation(self) -> pb.ResponseObservation:
        return self.send(observation=pb.RequestObservation()).observation

    def kill_unit(self, unit_tags: list[int]) -> pb.ResponseDebug:
        command = DebugCommand(kill_unit=DebugKillUnit(tag=unit_tags))
        return self.send(debug=pb.RequestDebug(debug=[command])).debug

    def create_unit(
        self, unit_type: int, owner: int, pos: Point2D, quantity: int = 1
    ) -> pb.ResponseDebug:
        create = DebugCreateUnit(
            unit_type=unit_type, owner=owner, pos=pos, quantity=quantity
        )
        command = DebugCommand(create_unit=create)
        return self.send(debug=pb.RequestDebug(debug=[command])).debug

    def get_game_info(self) -> pb.ResponseGameInfo:
        return self.send(game_info=pb.RequestGameInfo()).game_info

    def get_game_data(self) -> pb.ResponseData:
        request = pb.RequestData(
            ability_id=True,
            unit_type_id=True,
            upgrade_id=True,
            buff_id=True,
            effect_id=True,
        )
        return self.send(data=request).data

    def join_game(self) -> pb.ResponseJoinGame:
        interface_options = pb.InterfaceOptions(
            raw=True,
            score=True,
            show_cloaked=True,
            show_placeholders=True,
            show_burrowed_shadows=True,
            raw_affects_selection=False,
            raw_crop_to_playable_area=False,
        )
        request = pb.RequestJoinGame(race=self.players[0], options=interface_options)
        response = self.send(join_game=request).join_game
        assert not response.HasField("error"), (
            f"JoinGame failed: {response.error_details}"
        )
        return response

    def send(self, **kwargs) -> pb.Response:
        request = pb.Request(**kwargs)
        self.socket.send_bytes(request.SerializeToString())
        response_raw = self.socket.recv()
        assert isinstance(response_raw, bytes)
        response = pb.Response()
        response.ParseFromString(response_raw)
        return response


class SC2Game(SC2BackgroundServer):
    """Processing layer: converts protobuf messages into usable formats."""

    def __init__(self) -> None:
        super().__init__()
        self._game_info = self.get_game_info()
        self._map_size = (
            self._game_info.start_raw.map_size.x,
            self._game_info.start_raw.map_size.y,
        )
        self._playable_area = self._game_info.start_raw.playable_area

    @staticmethod
    def unpack_grid(data: bytes, width: int, height: int) -> np.ndarray:
        """Unpack bit-packed grid data to numpy array."""
        buffer = np.frombuffer(data, dtype=np.uint8)
        buffer = np.unpackbits(buffer)
        expected_size = width * height
        if len(buffer) >= expected_size:
            buffer = buffer[:expected_size]
        else:
            buffer = np.pad(buffer, (0, expected_size - len(buffer)))
        return buffer.reshape(height, width)

    def get_random_position(self) -> Point2D:
        """Get a random position within the playable area."""
        p0 = self._playable_area.p0
        p1 = self._playable_area.p1
        x = random.uniform(p0.x + 5, p1.x - 5)
        y = random.uniform(p0.y + 5, p1.y - 5)
        return Point2D(x=x, y=y)

    def kill_all_units(self) -> None:
        """Kill all units on the map."""
        obs = self.observation()
        unit_tags = [unit.tag for unit in obs.observation.raw_data.units]
        if unit_tags:
            self.kill_unit(unit_tags)
            self.step()

    def spawn_units(self, unit_type: int, owner: int, quantity: int) -> None:
        """Spawn units at a random position."""
        pos = self.get_random_position()
        self.create_unit(unit_type=unit_type, owner=owner, pos=pos, quantity=quantity)
        self.step()

    def get_visibility_grid(self) -> np.ndarray:
        """Get visibility map as numpy array."""
        obs = self.observation()
        map_state = obs.observation.raw_data.map_state
        width, height = self._map_size
        return self.unpack_grid(map_state.visibility.data, width, height)

    def get_creep_grid(self) -> np.ndarray:
        """Get creep map as numpy array."""
        obs = self.observation()
        map_state = obs.observation.raw_data.map_state
        width, height = self._map_size
        return self.unpack_grid(map_state.creep.data, width, height)

    def get_unit_positions(self, owner: int) -> np.ndarray:
        """Get unit positions for a player as numpy array."""
        obs = self.observation()
        width, height = self._map_size
        grid = np.zeros((height, width), dtype=np.uint8)
        for unit in obs.observation.raw_data.units:
            if unit.owner == owner:
                x = int(unit.pos.x)
                y = int(unit.pos.y)
                if 0 <= x < width and 0 <= y < height:
                    grid[y, x] = 1
        return grid

    def get_pathing_grid(self) -> np.ndarray:
        """Get static pathing grid as numpy array."""
        pathing = self._game_info.start_raw.pathing_grid
        return self.unpack_grid(pathing.data, pathing.size.x, pathing.size.y)
