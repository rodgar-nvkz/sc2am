# AGENTS.MD

## Project Overview

SC2AM (StarCraft II Autonomous Model) is a high-throughput RL training framework for StarCraft II. It provides a PettingZoo-compatible parallel environment that communicates directly with SC2 via `s2clientprotocol` websockets.

## Tech Stack

- Python 3.14+
- `s2clientprotocol` - Direct SC2 API communication
- `pettingzoo` - Multi-agent RL environment interface
- `burnysc2` - Used in research scripts only
- `uv` - Package manager

## Structure

```
scam/
├── environments/
│   ├── base.py      # SC2BackgroundServer, SC2Game - raw SC2 connection & game control
│   └── minigame.py  # MinigameEnvironment - PettingZoo ParallelEnv implementation
├── maps/
│   └── generator.py # MapGenerator - loads/validates .SC2Map files
└── research/        # Experimental scripts
```

## Key Classes

- `SC2BackgroundServer`: Spawns SC2 process, manages websocket, sends protobuf requests
- `SC2Game`: Extends server with observation helpers (grids, unit positions)
- `MinigameEnvironment`: PettingZoo `ParallelEnv` wrapping SC2Game

## Notes

- SC2 must be installed at `~/StarCraftII/`
- Maps stored in `maps/` folder (MPQ format)
- Research folder contains reference implementations, not core functionality
