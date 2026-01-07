from dataclasses import dataclass

import websocket
from portpicker import PickUnusedPort
from s2clientprotocol import raw_pb2 as raw
from s2clientprotocol import sc2api_pb2 as pb
from s2clientprotocol.common_pb2 import Point2D, Race
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
    # stableid.json
    ABILITY_MOVE = 16
    ABILITY_ATTACK = 23
    ABILITY_STOP = 3665

    def __init__(self, socket: websocket.WebSocket) -> None:
        self.socket = socket

    @staticmethod
    def player(race: Race, controlled: bool) -> pb.PlayerSetup:
        type = pb.Participant if controlled else pb.Computer
        return pb.PlayerSetup(race=race, type=type)

    def send(self, **kwargs) -> pb.Response:
        request = pb.Request(**kwargs)
        self.socket.send_bytes(request.SerializeToString())
        response_raw = self.socket.recv()
        assert isinstance(response_raw, bytes)
        response = pb.Response()
        response.ParseFromString(response_raw)
        return response

    def host_game(self, map_data: bytes, players: list[pb.PlayerSetup]) -> pb.ResponseCreateGame:
        local_map = pb.LocalMap(map_data=map_data)
        request = pb.RequestCreateGame(
            local_map=local_map, player_setup=players, realtime=False
        )
        response = self.send(create_game=request).create_game
        return response

    def join_game(self, race: Race, port_config: PortConfig | None = None, host_ip: str | None = None) -> pb.ResponseJoinGame:
        interface_options = pb.InterfaceOptions(
            raw=True,
            score=True,
            show_cloaked=True,
            show_placeholders=True,
            show_burrowed_shadows=True,
            raw_affects_selection=True,
            raw_crop_to_playable_area=True,
        )
        request = pb.RequestJoinGame(race=race, options=interface_options)
        if port_config:
            request.shared_port = port_config.shared_port
            request.server_ports = port_config.server_port_set
            request.client_ports.extend(port_config.client_port_set)
        if host_ip is not None:
            request.host_ip = host_ip
        return self.send(join_game=request).join_game

    def restart_game(self) -> pb.ResponseRestartGame:
        """"Restart not supported in multiplayer, extremely slow in singleplayer"""
        return self.send(restart_game=pb.RequestRestartGame()).restart_game

    def step(self, count: int = 1) -> pb.ResponseStep:
        return self.send(step=pb.RequestStep(count=count)).step

    def get_observation(self, disable_fog: bool = True) -> pb.ResponseObservation:
        return self.send(observation=pb.RequestObservation(disable_fog=disable_fog)).observation

    def get_game_info(self) -> pb.ResponseGameInfo:
        return self.send(game_info=pb.RequestGameInfo()).game_info

    def get_game_data(self) -> pb.ResponseData:
        request = pb.RequestData(ability_id=True, unit_type_id=True, upgrade_id=True, buff_id=True, effect_id=True)
        return self.send(data=request).data

    def kill_units(self, unit_tags: list[int]) -> pb.ResponseDebug:
        if not unit_tags:
            return pb.ResponseDebug()
        command = DebugCommand(kill_unit=DebugKillUnit(tag=unit_tags))
        return self.send(debug=pb.RequestDebug(debug=[command])).debug

    def spawn_units(self, unit_type: int, pos: tuple[float, float], owner: int, quantity: int = 1) -> pb.ResponseDebug:
        position = Point2D(x=pos[0], y=pos[1])
        create = DebugCreateUnit(unit_type=unit_type, owner=owner, pos=position, quantity=quantity)
        command = DebugCommand(create_unit=create)
        return self.send(debug=pb.RequestDebug(debug=[command])).debug

    def unit_command(self, cmd: raw.ActionRawUnitCommand) -> pb.ResponseAction:
        action = pb.Action(action_raw=raw.ActionRaw(unit_command=cmd))
        return self.send(action=pb.RequestAction(actions=[action])).action

    def unit_move(self, unit_tag: int, target_pos: tuple[float, float]) -> pb.ResponseAction:
        cmd = raw.ActionRawUnitCommand(ability_id=self.ABILITY_MOVE, unit_tags=[unit_tag])
        cmd.target_world_space_pos.x, cmd.target_world_space_pos.y = target_pos
        return self.unit_command(cmd)

    def unit_stop(self, unit_tag: int) -> pb.ResponseAction:
        cmd = raw.ActionRawUnitCommand(ability_id=self.ABILITY_STOP, unit_tags=[unit_tag])
        return self.unit_command(cmd)

    def unit_attack(self, unit_tag: int, target_pos: tuple[float, float]) -> pb.ResponseAction:
        cmd = raw.ActionRawUnitCommand(ability_id=self.ABILITY_ATTACK, unit_tags=[unit_tag])
        cmd.target_world_space_pos.x, cmd.target_world_space_pos.y = target_pos
        return self.unit_command(cmd)

    def unit_attack_unit(self, unit_tag: int, target_unit_tag: int) -> pb.ResponseAction:
        cmd = raw.ActionRawUnitCommand(ability_id=self.ABILITY_ATTACK, unit_tags=[unit_tag])
        cmd.target_unit_tag = target_unit_tag
        return self.unit_command(cmd)

    def save_replay(self) -> bytes:
        return self.send(save_replay=pb.RequestSaveReplay()).save_replay.data


class SC2Client(SC2ClientProtocol):
    """SC2 Client over WebSocket connection"""
