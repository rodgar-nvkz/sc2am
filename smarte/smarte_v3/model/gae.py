"""Generalized Advantage Estimation (GAE) for variance reduction.

GAE provides a family of estimators parameterized by λ ∈ [0, 1]:
- λ = 0: TD(0) - low variance, high bias
- λ = 1: MC returns - high variance, low bias
- λ = 0.95: Good balance for most RL tasks

GAE computes advantages PER-EPISODE using exponentially-weighted
TD errors. This avoids the "averaging problem" where advantages from different
episodes get mixed during batch normalization.
"""

import numpy as np


def compute_gae_episode(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages and λ-returns for a single complete episode.

    GAE(λ) advantage at timestep t:
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

    where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error.

    The λ-return (used as value target) is:
        G_t^λ = A_t + V(s_t)

    Args:
        rewards: Rewards for each timestep (T,)
        values: Value estimates V(s_t) from behavior policy (T,)
        gamma: Discount factor (default: 0.99)
        gae_lambda: GAE λ parameter (default: 0.95)
            - Higher λ = less bias, more variance
            - Lower λ = more bias, less variance

    Returns:
        Tuple of:
        - lambda_returns: λ-return targets for value function (T,)
        - advantages: GAE advantages for policy gradient (T,)

    Note:
        Since this is a complete episode, the terminal state has V(s_T) = 0.
        We iterate backwards from the terminal state.
    """
    T = len(rewards)

    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0  # Accumulated GAE, starts at 0 from terminal

    # Iterate backwards from terminal state
    for t in reversed(range(T)):
        # Next value: 0 at terminal, otherwise V(s_{t+1})
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE accumulation: A_t = δ_t + γλ * A_{t+1}
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    # λ-returns: G_t^λ = A_t + V(s_t)
    # This is the target for the value function
    lambda_returns = advantages + values

    return lambda_returns, advantages
