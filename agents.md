# AGENTS.MD

## Project Overview
SMARTE (StarCraft II Multi Agent Reinforcement Learning Environment) is a high-throughput RL training framework for StarCraft II. It provides a PettingZoo-compatible parallel environment that communicates directly with SC2 via `s2clientprotocol` websockets.

## Current problem-to-solve definition
Full task: chase to zerglings, attack, kite if nearby to avoid damage having only win/loose signal.
Temporal hint: loose episodes provides a slight signal about full task progress, 1 - (enemy hp left), which ends with 1 - 0 = 1 in case of win.

Zerglins spawn positions are
1. In the range of sight: < 8. Attack command will lead to move-and-attack
2. Out of the range of sight: > 12. Attack command will do nothing, should ddo some moe ahead before
3. The only one way to win in a fight is fire-kite strategy, with attacs only marine died with 1 enemy killed and second one has half of hit points

The full task might be splitted into the next parts:
1. Learn to chase if far away
2. Learn to kite if nearby
3. Combine both properly to solve the full task in a ~100 environment steps

Notes:
- Since in this particular env\problem we are able to provide better intermidiate rewards, we will not in more complex scenarios. Our aim is not to solve concrete problem, but find a way to learn a model which might learn to solve complex scenarious on their own with a win/lose final signal and DPS signal during gameplay. Agent should be able to realise what muktiple DPS signals (e.g. burst in one case vs sustained damage in another) leads to the win avoiding local optimas.
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


## IMPALA Training (`smarte/impala_*/`)
Marine vs Zerglings trainer with hybrid actions (discrete command + continuous angle).
Async distributed workers collects complete episodes, learner batches them and trains. No fixed rollouts, no padding, no mini-batches.
Key insight: for sparse terminal rewards, match your abstractions to the problem - think in episodes, not steps.
Bigger batches with more epochs extract more signal. Watch staleness to know when you've pushed too far.

## SMARTE v0 (`smarte/smarte_v0/`)
LSTM-based discrete action space variant. Key improvements over impala_v2:

### Reward Strategies (`rewards.py`)
Configurable reward shaping to handle different learning challenges:

- **terminal_only**: Pure terminal reward (win/lose with partial credit). Best for simple tasks.
- **damage_only**: Reward damage dealt, no penalty for taking. Good for kiting.
- **time_penalty**: Terminal + small time penalty. **Recommended for chase+kite** - implicitly encourages engagement.
- **momentum**: DPS trend comparison over rolling window.
- **asymmetric**: Deal damage worth more than take damage penalty.
- **potential**: Potential-based shaping (theoretically optimal-policy preserving).

Configure via `config.reward_strategy` and `config.reward_params`.

### Chase Learning Problem
When enemies spawn far away (>12, out of sight), attack command does nothing → zero intermediate reward → model must randomly discover movement leads to combat.

**Solutions implemented:**
1. **Time penalty** (-0.001/step): Standing still wastes time, implicitly encourages approach
2. **Entropy annealing**: High entropy early (0.05→0.005) encourages exploration of movement options
3. **No damage-taken penalty**: Original `dealt - taken` reward penalized necessary damage intake

### Key Insight
The `dealt - taken` per-step reward is problematic because:
- Can't win without taking some damage (kiting isn't perfect)
- Zero signal during chase phase (no combat = no reward)
- Penalizes good play that involves calculated risk

impala_v2 solved this with terminal-only reward: `1 - (enemy_hp_left / max_hp)` at episode end.
