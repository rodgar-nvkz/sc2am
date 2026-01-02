import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import portpicker
import websocket
from loguru import logger
from s2clientprotocol import sc2api_pb2 as pb
from s2clientprotocol.common_pb2 import Point2D, Race
from s2clientprotocol.debug_pb2 import (
    DebugCommand,
    DebugCreateUnit,
    DebugEndGame,
    DebugKillUnit,
)

GAME_FOLDER = Path("~/StarCraftII/").expanduser()
GAME_ENTRYPOINT = Path("Versions/Base75689/SC2_x64")
SERVER_HOST = "127.0.0.1"


@dataclass
class SC2GameProp:
    map_path: str
    players: list[int]


class SC2BackgroundServer:
    def __init__(self, game: SC2GameProp) -> None:
        self.game = game
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
        self.host_game()
        response_join_game = self.join_game(self.game.players[0])
        return response_join_game.player_id

    def step(self) -> pb.ResponseStep:
        return self.send(step=pb.RequestStep(count=1)).step

    def observation(self) -> pb.ResponseObservation:
        return self.send(observation=pb.RequestObservation()).observation

    def kill_unit(self, unit_tags: list[int]) -> pb.ResponseDebug:
        command = DebugCommand(kill_unit=DebugKillUnit(tag=unit_tags))
        return self.send(debug=pb.RequestDebug(debug=[command])).debug

    def create_unit(
        self, unit_type: int, owner: int, x: float, y: float, quantity: int = 1
    ) -> pb.ResponseDebug:
        """Create units at a specified position."""
        pos = Point2D(x=x, y=y)
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

    def join_game(self, race: Race) -> pb.ResponseJoinGame:
        interface_options = pb.InterfaceOptions(
            raw=True,
            score=True,
            show_cloaked=True,
            show_placeholders=True,
            show_burrowed_shadows=True,
            raw_affects_selection=False,
            raw_crop_to_playable_area=False,
        )
        request = pb.RequestJoinGame(race=race, options=interface_options)
        response = self.send(join_game=request).join_game
        assert not response.HasField("error"), (
            f"JoinGame failed: {response.error_details}"
        )
        return response

    def host_game(self):
        players = [
            pb.PlayerSetup(
                race=self.game.players[0],
                type=pb.PlayerType.Participant,
            ),
            pb.PlayerSetup(
                race=self.game.players[0],
                type=pb.PlayerType.Computer,
            ),
        ]
        local_map = pb.LocalMap(map_path=self.game.map_path)
        request = pb.RequestCreateGame(
            local_map=local_map, player_setup=players, realtime=False
        )
        response = self.send(create_game=request).create_game
        assert not response.HasField("error"), response.error
        return response

    def send(self, **kwargs) -> pb.Response:
        request = pb.Request(**kwargs)
        self.socket.send_bytes(request.SerializeToString())
        response_raw = self.socket.recv()
        assert isinstance(response_raw, bytes)
        response = pb.Response()
        response.ParseFromString(response_raw)
        return response
