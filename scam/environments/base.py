import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import portpicker
import websocket
from loguru import logger
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ActionType, AgentID
from s2clientprotocol import sc2api_pb2
from s2clientprotocol.common_pb2 import Race

GAME_FOLDER = Path("~/StarCraftII/").expanduser()
GAME_ENTRYPOINT = Path("Versions/Base75689/SC2_x64")
SERVER_HOST = "127.0.0.1"


@dataclass
class SC2GameProp:
    map_path: str
    players: list[Race]


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

    def step(self) -> sc2api_pb2.ResponseStep:
        return self.send(step=sc2api_pb2.RequestStep(count=1)).step

    def observation(self) -> sc2api_pb2.ResponseObservation:
        return self.send(observation=sc2api_pb2.RequestObservation()).observation

    def get_game_info(self) -> sc2api_pb2.ResponseGameInfo:
        return self.send(game_info=sc2api_pb2.RequestGameInfo()).game_info

    def get_game_data(self) -> sc2api_pb2.ResponseData:
        request = sc2api_pb2.RequestData(
            ability_id=True,
            unit_type_id=True,
            upgrade_id=True,
            buff_id=True,
            effect_id=True,
        )
        return self.send(data=request).data

    def join_game(self, race: Race) -> sc2api_pb2.ResponseJoinGame:
        interface_options = sc2api_pb2.InterfaceOptions(
            raw=True,
            score=True,
            show_cloaked=True,
            show_placeholders=True,
            show_burrowed_shadows=True,
            raw_affects_selection=False,
            raw_crop_to_playable_area=False,
        )
        request = sc2api_pb2.RequestJoinGame(race=race, options=interface_options)
        response = self.send(join_game=request).join_game
        assert not response.HasField("error"), (
            f"JoinGame failed: {response.error_details}"
        )
        return response

    def host_game(self):
        players = [
            sc2api_pb2.PlayerSetup(
                race=self.game.players[0],
                type=sc2api_pb2.PlayerType.Participant,
            ),
            sc2api_pb2.PlayerSetup(
                race=self.game.players[0],
                type=sc2api_pb2.PlayerType.Computer,
            ),
        ]
        local_map = sc2api_pb2.LocalMap(map_path=self.game.map_path)
        request = sc2api_pb2.RequestCreateGame(
            local_map=local_map, player_setup=players, realtime=False
        )
        response = self.send(create_game=request).create_game
        assert not response.HasField("error"), response.error
        return response

    def send(self, **kwargs) -> sc2api_pb2.Response:
        request = sc2api_pb2.Request(**kwargs)
        self.socket.send_bytes(request.SerializeToString())
        response_raw = self.socket.recv()
        assert isinstance(response_raw, bytes)
        response = sc2api_pb2.Response()
        response.ParseFromString(response_raw)
        return response


class SC2Env(ParallelEnv):
    metadata = {"name": "sc2_env_v0"}

    def __init__(self) -> None:
        self.server = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        pass

    def step(self, actions: dict[AgentID, ActionType]):
        pass
