"""Remote SC2 client for connecting to an external SC2 server."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from random import randint

from aiohttp import ClientSession, ClientWebSocketResponse
from aiohttp.client_ws import ClientWSTimeout
from loguru import logger
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2.bot_ai import BotAI
from sc2.client import Client
from sc2.data import CreateGameError, Race, Result
from sc2.game_state import GameState
from sc2.player import AbstractPlayer, Bot, Computer
from sc2.protocol import ConnectionAlreadyClosedError, Protocol, ProtocolError


class RemoteController(Protocol):
    """Controller for a remote SC2 server (no local process)."""

    async def create_game(self, map_path: str, players: list[AbstractPlayer]):
        """Create a game on the remote server."""
        req = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(map_path=map_path),
            random_seed=randint(0, 2**32 - 1),
        )

        for player in players:
            p = req.player_setup.add()
            p.type = player.type.value
            if isinstance(player, Computer):
                p.race = player.race.value
                p.difficulty = player.difficulty.value
                p.ai_build = player.ai_build.value

        logger.info("Creating new game on remote server")
        logger.info(f"Map: {map_path}")
        logger.info(f"Players: {', '.join(str(p) for p in players)}")

        result = await self._execute(create_game=req)

        if result.create_game.HasField("error"):
            err = f"Could not create game: {CreateGameError(result.create_game.error)}"
            if result.create_game.HasField("error_details"):
                err += f": {result.create_game.error_details}"
            logger.critical(err)
            raise RuntimeError(err)

        return result


async def _play_game_ai(
    client: Client,
    player_id: int,
    ai: BotAI,
    realtime: bool,
    game_time_limit: int | None,
) -> Result:
    """Run the bot AI game loop."""
    gs: GameState | None = None

    async def initialize_first_step() -> Result | None:
        nonlocal gs
        ai._initialize_variables()

        game_data = await client.get_game_data()
        game_info = await client.get_game_info()
        ping_response = await client.ping()

        ai._prepare_start(
            client,
            player_id,
            game_info,
            game_data,
            realtime=realtime,
            base_build=ping_response.ping.base_build,
        )
        state = await client.observation()
        if client._game_result:
            await ai.on_end(client._game_result[player_id])
            return client._game_result[player_id]

        gs = GameState(state.observation)
        proto_game_info = await client._execute(game_info=sc_pb.RequestGameInfo())

        try:
            ai._prepare_step(gs, proto_game_info)
            await ai.on_before_start()
            ai._prepare_first_step()
            await ai.on_start()
        except Exception as e:
            logger.exception(f"Caught exception in AI on_start: {e}")
            logger.error("Resigning due to previous error")
            await ai.on_end(Result.Defeat)
            return Result.Defeat

        return None

    result = await initialize_first_step()
    if result is not None:
        return result

    async def run_bot_iteration(iteration: int):
        nonlocal gs
        logger.debug(f"Running AI step, it={iteration} {gs.game_loop / 22.4:.2f}s")
        await ai.issue_events()
        try:
            await ai.on_step(iteration)
        except Exception as e:
            logger.exception(f"Caught exception: {e}")
            raise
        await ai._after_step()
        logger.debug("Running AI step: done")

    previous_state_observation = None
    for iteration in range(10**10):
        if realtime and gs:
            from contextlib import suppress

            with suppress(ProtocolError):
                requested_step = gs.game_loop + client.game_step
                state = await client.observation(requested_step)
                if state.observation.observation.game_loop > requested_step:
                    logger.debug("Skipped a step in realtime=True")
                    previous_state_observation = state.observation
                    state = await client.observation(
                        state.observation.observation.game_loop + 1
                    )
        else:
            state = await client.observation()

        if client._game_result:
            await ai.on_end(client._game_result[player_id])
            return client._game_result[player_id]

        gs = GameState(state.observation, previous_state_observation)
        previous_state_observation = None
        logger.debug(f"Score: {gs.score.score}")

        if game_time_limit and gs.game_loop / 22.4 > game_time_limit:
            await ai.on_end(Result.Tie)
            return Result.Tie

        proto_game_info = await client._execute(game_info=sc_pb.RequestGameInfo())
        ai._prepare_step(gs, proto_game_info)

        await run_bot_iteration(iteration)

        if not realtime:
            if not client.in_game:
                await ai.on_end(client._game_result[player_id])
                return client._game_result[player_id]
            await client.step()

    return Result.Undecided


@dataclass
class RemoteGame:
    """Configuration for a remote game."""

    url: str
    map_path: str
    bot: Bot
    opponent: AbstractPlayer
    realtime: bool = False
    random_seed: int | None = None
    disable_fog: bool | None = None
    save_replay_as: str | None = None
    game_time_limit: int | None = None


async def run_game_remote(
    url: str,
    map_path: str,
    bot: Bot,
    opponent: AbstractPlayer,
    realtime: bool = False,
    random_seed: int | None = None,
    disable_fog: bool = False,
    game_time_limit: int = 60 * 5,
) -> Result | None:
    """Run a game on a remote SC2 server.

    This connects to an already-running SC2 instance, creates a game,
    joins it, and plays until completion.

    Args:
        url: WebSocket URL of the SC2 server (e.g., "ws://192.168.42.122:40123/sc2api")
        map_path: Path to the map on the remote server
        bot: Your bot player
        opponent: The opponent (Computer or another Bot)
        realtime: Whether to run in realtime mode
        random_seed: Optional random seed for the game
        disable_fog: Whether to disable fog of war
        save_replay_as: Optional path to save replay
        game_time_limit: Optional game time limit in seconds

    Returns:
        Result of the game (Victory, Defeat, Tie, etc.) or None on error
    """
    session: ClientSession | None = None
    ws: ClientWebSocketResponse | None = None

    try:
        logger.info(f"Connecting to remote SC2 server at {url}")
        session = ClientSession()
        ws = await session.ws_connect(url)

        # Create controller to set up the game
        controller = RemoteController(ws)
        await controller.ping()
        logger.info("Connected to remote server, ping successful")

        # Create the game
        await controller.create_game(
            map_path=map_path,
            players=[bot, opponent],
            realtime=realtime,
            random_seed=random_seed,
            disable_fog=disable_fog,
        )

        # Create client and join the game
        client = Client(
            ws, save_replay_path=f"./replays/{datetime.now().isoformat()}.SC2Replay"
        )
        player_id = await client.join_game(bot.name, bot.race)
        logger.info(f"Joined game as player {player_id}")

        result = await _play_game_ai(
            client, player_id, bot.ai, realtime, game_time_limit
        )
        await client.leave()

        logger.info(f"Game finished with result: {result}")
        return result

    except Exception as e:
        logger.exception(f"Error during remote game: {e}")
        return None

    finally:
        if ws is not None:
            await ws.close()
        if session is not None:
            await session.close()
