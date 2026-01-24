# AGENTS.MD

## Project Overview
SMARTE (StarCraft II Multi Agent Reinforcement Learning Environment) is a high-throughput RL training framework for StarCraft II. It provides a PettingZoo-compatible parallel environment that communicates directly with SC2 via `s2clientprotocol` websockets.

Mindset:
- Our aim is not to solve concrete env problem, but find a way to train a model which might learn to solve complex scenarious on their own with a win/lose final signal only.
- We would like to avoid adding more handcrafted features into the obs, keeping only raw game ones, since feature mining is also not possible for complex scenarious.

## Code formatting
- Always use one line functions and methods parameters even for a long one, linter will do the rest.

## Tech Stack
- Python 3.14+
- `s2clientprotocol` - Direct SC2 API communication
- `pettingzoo` - Multi-agent RL environment interface
- `uv` - Package manager
- `PYTHONPATH=. uv run python -m pytest` to run tests

## Key Classes
- `SC2BackgroundServer`: Spawns SC2 process, manages websocket, sends protobuf requests
- `SC2Game`: Extends server with observation helpers (grids, unit positions)
