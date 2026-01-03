from dataclasses import dataclass

import numpy as np
import websocket
from portpicker import PickUnusedPort
from s2clientprotocol import sc2api_pb2 as pb
from s2clientprotocol.common_pb2 import Point2D
from s2clientprotocol.debug_pb2 import DebugCommand, DebugCreateUnit, DebugKillUnit


@dataclass
class PortConfig:
    shared_port: int
    server_port_set: pb.PortSet
    client_port_set: list[pb.PortSet]

    @classmethod
    def pick_ports(cls, players_count: int) -> "PortConfig":
        return cls(
            shared_port=PickUnusedPort(),
            server_port_set=cls.port_set(),
            client_port_set=[cls.port_set() for _ in range(players_count - 1)],
        )

    @classmethod
    def port_set(cls):
        return pb.PortSet(game_port=PickUnusedPort(), base_port=PickUnusedPort())


class SC2ClientProtocol:
    def __init__(self, socket: websocket.WebSocket) -> None:
        self.socket = socket

    def send(self, **kwargs) -> pb.Response:
        request = pb.Request(**kwargs)
        self.socket.send_bytes(request.SerializeToString())
        response_raw = self.socket.recv()
        assert isinstance(response_raw, bytes)
        response = pb.Response()
        response.ParseFromString(response_raw)
        return response

    def host_game(self, map_data: bytes, players: list[int]):
        setup = [pb.PlayerSetup(race=race, type=pb.PlayerType.Participant) for race in players]
        local_map = pb.LocalMap(map_data=map_data)
        request = pb.RequestCreateGame(local_map=local_map, player_setup=setup, realtime=False)
        response = self.send(create_game=request).create_game
        return response

    def join_game(self, race: int, port_config: PortConfig, host_ip: str | None = None) -> pb.ResponseJoinGame:
        interface_options = pb.InterfaceOptions(
            raw=True,
            score=True,
            show_cloaked=True,
            show_placeholders=False,
            show_burrowed_shadows=True,
            raw_affects_selection=False,
            raw_crop_to_playable_area=False,
        )
        request = pb.RequestJoinGame(
            race=race,
            options=interface_options,
            server_ports=port_config.server_port_set,
            client_ports=port_config.client_port_set,
            shared_port=port_config.shared_port,
        )
        if host_ip is not None:
            request.host_ip = host_ip

        response = self.send(join_game=request).join_game
        return response

    def step(self, count: int = 1) -> pb.ResponseStep:
        return self.send(step=pb.RequestStep(count=count)).step

    def get_observation(self) -> pb.ResponseObservation:
        return self.send(observation=pb.RequestObservation()).observation

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

    def kill_unit(self, unit_tags: list[int]) -> pb.ResponseDebug:
        command = DebugCommand(kill_unit=DebugKillUnit(tag=unit_tags))
        return self.send(debug=pb.RequestDebug(debug=[command])).debug

    def create_unit(self, unit_type: int, owner: int, pos: Point2D, quantity: int = 1) -> pb.ResponseDebug:
        create = DebugCreateUnit(unit_type=unit_type, owner=owner, pos=pos, quantity=quantity)
        command = DebugCommand(create_unit=create)
        return self.send(debug=pb.RequestDebug(debug=[command])).debug


class SC2Client(SC2ClientProtocol):
    def __init__(self, socket: websocket.WebSocket) -> None:
        super().__init__(socket)

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

    # def get_random_position(self) -> Point2D:
    #     """Get a random position within the playable area."""
    #     p0 = self._playable_area.p0
    #     p1 = self._playable_area.p1
    #     x = random.uniform(p0.x + 5, p1.x - 5)
    #     y = random.uniform(p0.y + 5, p1.y - 5)
    #     return Point2D(x=x, y=y)

    # def kill_all_units(self) -> None:
    #     """Kill all units on the map."""
    #     obs = self.observation()
    #     unit_tags = [unit.tag for unit in obs.observation.raw_data.units]
    #     if unit_tags:
    #         self.kill_unit(unit_tags)
    #         self.step()

    # def spawn_units(self, unit_type: int, owner: int, quantity: int) -> None:
    #     """Spawn units at a random position."""
    #     pos = self.get_random_position()
    #     self.create_unit(unit_type=unit_type, owner=owner, pos=pos, quantity=quantity)
    #     self.step()

    # def get_visibility_grid(self) -> np.ndarray:
    #     """Get visibility map as numpy array."""
    #     obs = self.observation()
    #     map_state = obs.observation.raw_data.map_state
    #     width, height = self._map_size
    #     return self.unpack_grid(map_state.visibility.data, width, height)

    # def get_creep_grid(self) -> np.ndarray:
    #     """Get creep map as numpy array."""
    #     obs = self.observation()
    #     map_state = obs.observation.raw_data.map_state
    #     width, height = self._map_size
    #     return self.unpack_grid(map_state.creep.data, width, height)

    # def get_unit_positions(self, owner: int) -> np.ndarray:
    #     """Get unit positions for a player as numpy array."""
    #     obs = self.observation()
    #     width, height = self._map_size
    #     grid = np.zeros((height, width), dtype=np.uint8)
    #     for unit in obs.observation.raw_data.units:
    #         if unit.owner == owner:
    #             x = int(unit.pos.x)
    #             y = int(unit.pos.y)
    #             if 0 <= x < width and 0 <= y < height:
    #                 grid[y, x] = 1
    #     return grid

    # def get_pathing_grid(self) -> np.ndarray:
    #     """Get static pathing grid as numpy array."""
    #     pathing = self._game_info.start_raw.pathing_grid
    #     return self.unpack_grid(pathing.data, pathing.size.x, pathing.size.y)
