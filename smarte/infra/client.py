from dataclasses import dataclass

import websocket
from loguru import logger
from portpicker import PickUnusedPort
from s2clientprotocol import raw_pb2 as raw
from s2clientprotocol import sc2api_pb2 as pb
from s2clientprotocol.common_pb2 import Point, Point2D, Race
from s2clientprotocol.debug_pb2 import (
    DebugCommand,
    DebugCreateUnit,
    DebugKillUnit,
    DebugSetScore,
    DebugSetUnitValue,
)


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
        self._pending_responses = 0

    @staticmethod
    def player(race: Race, controlled: bool) -> pb.PlayerSetup:
        type = pb.Participant if controlled else pb.Computer
        return pb.PlayerSetup(race=race, type=type)

    def send_nowait(self, **kwargs) -> None:
        """Fire request without waiting for response. Response will be drained on next send()."""
        request = pb.Request(**kwargs)
        self.socket.send_bytes(request.SerializeToString())
        self._pending_responses += 1

    def send(self, **kwargs) -> pb.Response:
        """Send request and return response. Drains any pending responses first."""
        request = pb.Request(**kwargs)
        self.socket.send_bytes(request.SerializeToString())

        while self._pending_responses > 0:
            self.socket.recv()
            self._pending_responses -= 1

        response = pb.Response()
        response.ParseFromString(self.socket.recv())  # type: ignore
        assert not response.error, f"SC2 API Error: {response.error}"
        return response

    def host_game(self, map_path: str, players: list[pb.PlayerSetup]) -> pb.ResponseCreateGame:
        # local_map = pb.LocalMap(map_data=map_data)
        local_map = pb.LocalMap(map_path=map_path)
        request = pb.RequestCreateGame(local_map=local_map, player_setup=players, realtime=False, disable_fog=True)
        return self.send(create_game=request).create_game

    def join_game(self, race: Race, port_config: PortConfig | None, host_ip: str | None) -> pb.ResponseJoinGame:
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
            request.server_ports.CopyFrom(port_config.server_port_set)
            request.client_ports.extend(port_config.client_port_set)
        if host_ip is not None:
            request.host_ip = host_ip
        return self.send(join_game=request).join_game

    def restart_game(self) -> pb.ResponseRestartGame:
        """Restart not supported in multiplayer, extremely slow in singleplayer"""
        return self.send(restart_game=pb.RequestRestartGame()).restart_game

    def step(self, count: int = 1) -> None:
        self.send_nowait(step=pb.RequestStep(count=count))

    def get_observation(self, disable_fog: bool = True) -> pb.ResponseObservation:
        return self.send(observation=pb.RequestObservation(disable_fog=disable_fog)).observation

    def get_game_info(self) -> pb.ResponseGameInfo:
        return self.send(game_info=pb.RequestGameInfo()).game_info

    def get_game_data(self) -> pb.ResponseData:
        request = pb.RequestData(ability_id=True, unit_type_id=True, upgrade_id=True, buff_id=True, effect_id=True)
        return self.send(data=request).data

    def kill_units(self, unit_tags: list[int]) -> None:
        if not unit_tags:
            return
        command = DebugCommand(kill_unit=DebugKillUnit(tag=unit_tags))
        self.send_nowait(debug=pb.RequestDebug(debug=[command]))

    def research_upgrades(self) -> None:
        """Calling multiple times unlocks progressive upgrades (e.g., +1, then +2, then +3)"""
        command = DebugCommand(game_state="upgrade")
        self.send_nowait(debug=pb.RequestDebug(debug=[command]))

    def set_unit_life(self, unit_tag: int, life: float) -> None:
        """Set unit health to a specific value (can exceed max health)"""
        value = DebugSetUnitValue(unit_tag=unit_tag, unit_value=DebugSetUnitValue.Life, value=life)
        command = DebugCommand(unit_value=value)
        self.send_nowait(debug=pb.RequestDebug(debug=[command]))

    def enable_enemy_control(self) -> None:
        """Enable control of enemy units (allows issuing commands to opponent's units)"""
        logger.debug("Enemy control enabled")
        command = DebugCommand(game_state="control_enemy")
        self.send_nowait(debug=pb.RequestDebug(debug=[command]))

    def spawn_units(self, unit_type: int, pos: tuple[float, float], owner: int, quantity: int = 1) -> None:
        position = Point2D(x=pos[0], y=pos[1])
        create = DebugCreateUnit(unit_type=unit_type, owner=owner, pos=position, quantity=quantity)
        command = DebugCommand(create_unit=create)
        self.send_nowait(debug=pb.RequestDebug(debug=[command]))

    def map_command(self, trigger_cmd: str) -> None:
        """Even supported in pb schema, this is a dead feature in Linux headless server (Base75689), always returns NoTriggerError.
        Binary analysis of SC2_x64 shows the handler at VA 0x105d980 only recognizes "reset" (game loop restart, VA 0x2fc7670),
        but even that is gated behind disabled feature flags (VA 0x1038948, 0x1038951)"""
        map_command = pb.RequestMapCommand(trigger_cmd=trigger_cmd)
        self.send_nowait(map_command=map_command)

    def unit_command(self, cmd: raw.ActionRawUnitCommand) -> None:
        action = pb.Action(action_raw=raw.ActionRaw(unit_command=cmd))
        self.send_nowait(action=pb.RequestAction(actions=[action]))

    def unit_move(self, unit_tag: int, target_pos: tuple[float, float]) -> None:
        cmd = raw.ActionRawUnitCommand(ability_id=self.ABILITY_MOVE, unit_tags=[unit_tag])
        cmd.target_world_space_pos.x, cmd.target_world_space_pos.y = target_pos
        self.unit_command(cmd)

    def unit_stop(self, unit_tag: int) -> None:
        cmd = raw.ActionRawUnitCommand(ability_id=self.ABILITY_STOP, unit_tags=[unit_tag])
        self.unit_command(cmd)

    def unit_attack(self, unit_tag: int, target_pos: tuple[float, float]) -> None:
        cmd = raw.ActionRawUnitCommand(ability_id=self.ABILITY_ATTACK, unit_tags=[unit_tag])
        cmd.target_world_space_pos.x, cmd.target_world_space_pos.y = target_pos
        self.unit_command(cmd)

    def unit_attack_unit(self, unit_tag: int, target_unit_tag: int) -> None:
        cmd = raw.ActionRawUnitCommand(
            ability_id=self.ABILITY_ATTACK, unit_tags=[unit_tag], target_unit_tag=target_unit_tag
        )
        self.unit_command(cmd)

    def save_replay(self) -> bytes:
        return self.send(save_replay=pb.RequestSaveReplay()).save_replay.data

    def set_score(self, value: float) -> None:
        """Set the custom score value (can be read by map triggers via c_playerPropCustom)"""
        command = DebugCommand(score=DebugSetScore(score=value))
        self.send_nowait(debug=pb.RequestDebug(debug=[command]))

    def send_chat(self, message: str) -> None:
        """Send a chat message (visible in replay)"""
        chat = pb.ActionChat(channel=pb.ActionChat.Broadcast, message=message)
        action = pb.Action(action_chat=chat)
        self.send_nowait(action=pb.RequestAction(actions=[action]))

    def camera_move(self, x: float, y: float) -> None:
        """Move camera to position (may trigger TriggerAddEventCameraMove in map)"""
        point = Point(x=x, y=y)
        action = pb.Action(action_raw=raw.ActionRaw(camera_move=raw.ActionRawCameraMove(center_world_space=point)))
        self.send_nowait(action=pb.RequestAction(actions=[action]))


class SC2Client(SC2ClientProtocol):
    """SC2 Client over WebSocket connection"""
