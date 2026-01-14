import torch


def compute_vtrace(
    behavior_log_probs: torch.Tensor,  # (B, T) combined log probs
    target_log_probs: torch.Tensor,    # (B, T) combined log probs
    rewards: torch.Tensor,             # (B, T)
    values: torch.Tensor,              # (B, T) current value estimates
    bootstrap_value: torch.Tensor,     # (B,) value at T+1
    dones: torch.Tensor,               # (B, T)
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute V-trace targets and advantages.

    Returns:
        vtrace_targets: (B, T) V-trace value targets
        advantages: (B, T) V-trace advantages for policy gradient
    """
    B, T = rewards.shape
    device = rewards.device

    # Compute importance sampling ratios
    log_rhos = target_log_probs - behavior_log_probs
    rhos = torch.exp(log_rhos)

    # Truncate ratios
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    # Compute TD errors: δ_t = ρ_t * (r_t + γ * V(s_{t+1}) - V(s_t))
    # Handle terminal states
    not_done = (~dones).float()

    # Shift values for V(s_{t+1})
    next_values = torch.zeros_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1] = bootstrap_value

    # Mask next values at episode boundaries
    next_values = next_values * not_done

    # TD errors
    deltas = clipped_rhos * (rewards + gamma * next_values - values)

    # Compute V-trace targets using backward recursion
    # v_s - V(s) = δ_s + γ * c_s * (v_{s+1} - V(s_{s+1}))
    vtrace_minus_v = torch.zeros(B, T + 1, device=device)

    for t in reversed(range(T)):
        vtrace_minus_v[:, t] = deltas[:, t] + gamma * cs[:, t] * not_done[:, t] * vtrace_minus_v[:, t + 1]

    vtrace_targets = vtrace_minus_v[:, :-1] + values

    # Advantages for policy gradient: ρ_t * (r_t + γ * v_{t+1} - V(s_t))
    # Using V-trace targets for v_{t+1}
    vtrace_next = torch.zeros_like(values)
    vtrace_next[:, :-1] = vtrace_targets[:, 1:]
    vtrace_next[:, -1] = bootstrap_value
    vtrace_next = vtrace_next * not_done

    advantages = clipped_rhos * (rewards + gamma * vtrace_next - values)

    return vtrace_targets, advantages
