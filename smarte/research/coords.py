"""Coordinate embedding research experiment.

Investigates whether a model can learn to compute distance/angle from raw (x, y)
coordinates via a shared point encoder, eliminating the need to precompute
distance/angle per each ally-enemy pair in observations (see smarte_v4/obs.py).

=============================================================================
MOTIVATION
=============================================================================
The current obs.py precomputes angle/distance for each ally-enemy pair and feeds
them directly to the model. This does not scale with unit count (N allies x M
enemies = N*M precomputed pairs). Instead, we want to feed raw (x, y) coords
and let the model compute geometry internally.

=============================================================================
EXPERIMENT PROGRESSION
=============================================================================

1. BASELINE (flat encoder, predict closest point):
   - Input: 5 points flattened (10 values) -> encoder -> task head predicts
     distance/angle to closest point from a query.
   - Result: Aux head 4.19 deg, Task head 56.90 deg (10k steps).
   - Problem: implicit argmin (find closest) is too hard for a small network.

2. TASK REFORMULATION (predict all points, not closest):
   - Task head predicts distance/angle to each of the 5 points from query.
   - Result: Aux 4.71 deg, Task 5.90 deg (10k steps).
   - Conclusion: removing argmin makes the task tractable.

3. SHARED POINT ENCODER (key architectural insight):
   - Single encoder (x,y) -> embedding, shared across all points and query.
   - Aux/task heads receive concatenated embeddings.
   - Result: Aux 4.25 deg, Task 3.28 deg (10k steps).
   - Conclusion: shared embedding is the right structure. Task head is now
     BETTER than aux because it has fewer outputs (5 vs 20 pairs).

4. ADDED RANGE (in_range classification):
   - Task head also receives a range float and predicts binary in_range flag.
   - Result: Aux 3.93 deg, Task 3.00 deg, In-range acc 93.7% (10k steps).

5. VARIABLE POINT COUNT (validity mask):
   - Points represented as (x, y, valid). 2-5 active points per sample.
   - Masked loss: only valid pairs/points contribute to gradients.
   - Result: Aux 3.87 deg, Task 2.89 deg, In-range acc 94.8% (10k steps).

6. PRECISION SCALING (architecture search, 50k steps):
   +--------------------------+----------+-----------+------------+--------+
   | Architecture             | Dist MAE | Angle MAE | Range Acc  | Params |
   +--------------------------+----------+-----------+------------+--------+
   | 3 layers, 32 hidden      | 0.0294   | 3.72 deg  | 96.8%      | ~10k   |
   | 2 layers, 64 hidden      | 0.0152   | 2.50 deg  | 94.9%      | ~18k   |
   | 3L enc + 2L heads, 64h   | 0.0197   | 2.54 deg  | 96.5%      | ~30k   |
   | 3 layers, 64 hidden      | 0.0100   | 1.59 deg  | 99.1%      | ~40k   |
   | 3 layers, 128 hidden     | 0.0091   | 0.94 deg  | 99.0%      | ~110k  |
   +--------------------------+----------+-----------+------------+--------+

=============================================================================
CONCLUSIONS
=============================================================================

1. Shared point encoder (x,y,valid) -> embedding is the correct architecture.
   Same encoder for ally/enemy/query points. Heads operate on embeddings.

2. Minimal viable architecture: 3 layers, 64 hidden, embed_dim=32.
   Achieves 0.01 normalized distance MAE and 1.6 deg angle error.

3. Precision vs map scale: 0.01 normalized error on a 256-unit map = 2.56
   game units error (half a marine attack range). NOT sufficient for precise
   micro. Solution: tile the map into 32x32 regions, run encoder per tile.
   0.01 error on 32-unit tile = 0.32 game units -- well within precision.

4. The approach naturally handles variable unit counts via validity flags,
   no architectural changes needed.

5. Aux pairwise prediction task helps the encoder learn spatial structure
   and should be retained during RL training.

=============================================================================
SUGGESTED ARCHITECTURE FOR PRODUCTION
=============================================================================

    point_encoder (shared weights):
        Linear(3, 64) -> SiLU -> Linear(64, 64) -> SiLU -> Linear(64, 32)

    Input per tile: up to N points as (x_local, y_local, valid)
    - x_local, y_local normalized to [0, 1] within the 32x32 tile
    - valid = 1.0 for alive units, 0.0 for empty slots

    aux_head (training signal):
        concat(all N embeddings) -> Linear -> SiLU -> Linear -> SiLU -> Linear
        Output: pairwise (distance, sin, cos) for all directed pairs
        Loss: masked MSE (only valid pairs)

    task_head (policy input):
        concat(all N embeddings, query_embedding, range) -> Linear -> SiLU
            -> Linear -> SiLU -> Linear
        Output: per-point (distance, sin, cos, in_range)

=============================================================================
CURRENT CODE STATE
=============================================================================
The code below implements experiment step 5 (variable count + range) with the
architecture from step 6 (configurable layers/hidden). Defaults set to the
best model (embed_dim=32, hidden=64, 3 layers) enougth to solve the problem.

Run: python smarte/research/coords.py
"""

import math

import torch
import torch.nn as nn

# =============================================================================
# Data generation
# =============================================================================


def generate_batch(batch_size: int, n_points: int = 5, min_active: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random points with variable number of active points.

    Each sample has between min_active and n_points active points.
    Inactive points are zeroed out and marked with valid=0.

    Returns:
        points: (batch_size, n_points, 3) tensor: [x, y, valid]
        n_active: (batch_size,) number of active points per sample
    """
    coords = torch.rand(batch_size, n_points, 2)
    n_active = torch.randint(min_active, n_points + 1, (batch_size,))
    valid = torch.zeros(batch_size, n_points, 1)
    for i in range(batch_size):
        valid[i, : n_active[i]] = 1.0
    coords = coords * valid
    points = torch.cat([coords, valid], dim=-1)  # (B, N, 3)
    return points, n_active


def compute_pairwise_targets(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute distance and angle for all directed pairs (valid pairs only).

    Args:
        points: (B, N, 3) tensor: [x, y, valid]

    Returns:
        targets: (B, N*(N-1), 3) [distance, sin(angle), cos(angle)] per pair
        mask: (B, N*(N-1)) 1.0 if both points valid, 0.0 otherwise
    """
    B, N, _ = points.shape
    coords = points[:, :, :2]
    valid = points[:, :, 2]  # (B, N)
    targets = []
    masks = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = coords[:, j, 0] - coords[:, i, 0]
            dy = coords[:, j, 1] - coords[:, i, 1]
            dist = torch.sqrt(dx * dx + dy * dy)
            angle = torch.atan2(dy, dx)
            pair_valid = valid[:, i] * valid[:, j]
            targets.append(torch.stack([dist, torch.sin(angle), torch.cos(angle)], dim=-1))
            masks.append(pair_valid)

    return torch.stack(targets, dim=1), torch.stack(masks, dim=1)


def compute_query_targets(
    points: torch.Tensor, query: torch.Tensor, range_val: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute distance + angle + in_range from query to each point (valid only).

    Args:
        points: (B, N, 3) tensor: [x, y, valid]
        query: (B, 2) query point
        range_val: (B,) attack range threshold

    Returns:
        targets: (B, N, 4) [distance, sin(angle), cos(angle), in_range]
        mask: (B, N) 1.0 if point valid, 0.0 otherwise
    """
    coords = points[:, :, :2]
    valid = points[:, :, 2]
    dx = coords[:, :, 0] - query[:, 0:1]
    dy = coords[:, :, 1] - query[:, 1:2]
    dist = torch.sqrt(dx * dx + dy * dy)
    angle = torch.atan2(dy, dx)
    in_range = (dist < range_val.unsqueeze(1)).float()
    targets = torch.stack([dist, torch.sin(angle), torch.cos(angle), in_range], dim=-1)
    return targets, valid


# =============================================================================
# Model
# =============================================================================


class CoordModel(nn.Module):
    """Shared point embedding model with validity-aware inputs.

    point_encoder: (x, y, valid) -> embedding (shared for all points and query)
    Aux head: concat(N embeddings) -> pairwise distance/angle
    Task head: concat(N embeddings, query_embedding, range) -> per-point distance/angle/in_range
    """

    def __init__(self, n_points: int = 5, embed_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.n_points = n_points
        n_pairs = n_points * (n_points - 1)
        aux_output = n_pairs * 3
        task_output = n_points * 4  # distance, sin, cos, in_range

        # Shared point encoder: (x, y, valid) -> embed_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, embed_dim),
        )

        # Aux head: all point embeddings -> pairwise geometry
        self.aux_head = nn.Sequential(
            nn.Linear(n_points * embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, aux_output),
        )

        # Task head: all point embeddings + query embedding + range -> per-point geometry
        self.task_head = nn.Sequential(
            nn.Linear((n_points + 1) * embed_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, task_output),
        )

    def forward(
        self, points: torch.Tensor, query: torch.Tensor, range_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            points: (B, N, 3) [x, y, valid]
            query: (B, 2) query point
            range_val: (B,) attack range threshold

        Returns:
            aux_pred: (B, n_pairs * 3)
            task_pred: (B, n_points * 4)
        """
        B, N, _ = points.shape

        # Embed all points with shared encoder (receives x, y, valid)
        all_embeds = self.point_encoder(points)  # (B, N, embed_dim)
        all_embeds_flat = all_embeds.flatten(1)  # (B, N * embed_dim)
        aux_pred = self.aux_head(all_embeds_flat)

        # Query gets valid=1 always
        query_with_valid = torch.cat([query, torch.ones(B, 1, device=query.device)], dim=-1)  # (B, 3)
        query_embed = self.point_encoder(query_with_valid)  # (B, embed_dim)
        task_input = torch.cat([all_embeds_flat, query_embed, range_val.unsqueeze(-1)], dim=-1)
        task_pred = self.task_head(task_input)

        return aux_pred, task_pred


# =============================================================================
# Training
# =============================================================================


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss computed only on valid (masked) elements.

    Args:
        pred: (B, N, D) predictions
        target: (B, N, D) targets
        mask: (B, N) validity mask (1=valid, 0=ignore)
    """
    mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
    sq_err = (pred - target) ** 2 * mask_expanded
    n_valid = mask_expanded.sum().clamp(min=1.0)
    return sq_err.sum() / n_valid


def train():
    n_points = 5
    batch_size = 256
    n_steps = 50_000
    lr = 1e-3
    log_every = 500

    model = CoordModel(n_points=n_points)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {n_steps} steps, batch_size={batch_size}")
    print()

    for step in range(1, n_steps + 1):
        points, _ = generate_batch(batch_size, n_points)
        query = torch.rand(batch_size, 2)
        range_val = torch.rand(batch_size) * 0.5 + 0.1  # [0.1, 0.6]

        # Targets
        aux_target, aux_mask = compute_pairwise_targets(points)  # (B,20,3), (B,20)
        task_target, task_mask = compute_query_targets(points, query, range_val)  # (B,5,4), (B,5)

        # Forward
        aux_pred, task_pred = model(points, query, range_val)

        # Reshape predictions for masked loss
        aux_pred_r = aux_pred.view(batch_size, -1, 3)
        task_pred_r = task_pred.view(batch_size, n_points, 4)

        # Losses (only on valid pairs/points)
        aux_loss = masked_mse_loss(aux_pred_r, aux_target, aux_mask)
        task_loss = masked_mse_loss(task_pred_r, task_target, task_mask)
        loss = aux_loss + task_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            print(f"step {step:>5d} | loss {loss.item():.6f} | aux {aux_loss.item():.6f} | task {task_loss.item():.6f}")

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    evaluate(model, n_points)


# =============================================================================
# Evaluation
# =============================================================================


def evaluate(model: CoordModel, n_points: int, n_samples: int = 1024):
    model.eval()
    with torch.no_grad():
        points, _ = generate_batch(n_samples, n_points)
        query = torch.rand(n_samples, 2)
        range_val = torch.rand(n_samples) * 0.5 + 0.1

        aux_target, aux_mask = compute_pairwise_targets(points)
        task_target, task_mask = compute_query_targets(points, query, range_val)

        aux_pred, task_pred = model(points, query, range_val)

        # Aux metrics (only valid pairs)
        aux_pred_r = aux_pred.view(n_samples, -1, 3)
        valid_aux = aux_mask.unsqueeze(-1).bool().expand_as(aux_pred_r)

        dist_err = (aux_pred_r[:, :, 0] - aux_target[:, :, 0]).abs()
        dist_err = (dist_err * aux_mask).sum() / aux_mask.sum()

        pred_angle = torch.atan2(aux_pred_r[:, :, 1], aux_pred_r[:, :, 2])
        true_angle = torch.atan2(aux_target[:, :, 1], aux_target[:, :, 2])
        angle_diff = (pred_angle - true_angle + math.pi) % (2 * math.pi) - math.pi
        angle_err = (angle_diff.abs() * aux_mask).sum() / aux_mask.sum()
        angle_err_deg = angle_err * 180.0 / math.pi

        print(f"\nAux head (pairwise, valid pairs only):")
        print(f"  Distance MAE: {dist_err.item():.6f} (normalized units)")
        print(f"  Angle MAE:    {angle_err_deg.item():.2f} degrees")

        # Task metrics (only valid points)
        task_pred_r = task_pred.view(n_samples, n_points, 4)

        t_dist_err = (task_pred_r[:, :, 0] - task_target[:, :, 0]).abs()
        t_dist_err = (t_dist_err * task_mask).sum() / task_mask.sum()

        t_pred_angle = torch.atan2(task_pred_r[:, :, 1], task_pred_r[:, :, 2])
        t_true_angle = torch.atan2(task_target[:, :, 1], task_target[:, :, 2])
        t_angle_diff = (t_pred_angle - t_true_angle + math.pi) % (2 * math.pi) - math.pi
        t_angle_err = (t_angle_diff.abs() * task_mask).sum() / task_mask.sum()
        t_angle_err_deg = t_angle_err * 180.0 / math.pi

        # In-range accuracy (only valid points)
        in_range_pred = (task_pred_r[:, :, 3] > 0.5).float()
        in_range_true = task_target[:, :, 3]
        in_range_correct = ((in_range_pred == in_range_true).float() * task_mask).sum()
        in_range_acc = in_range_correct / task_mask.sum()

        print(f"\nTask head (query -> each point, valid only):")
        print(f"  Distance MAE:    {t_dist_err.item():.6f} (normalized units)")
        print(f"  Angle MAE:       {t_angle_err_deg.item():.2f} degrees")
        print(f"  In-range acc:    {in_range_acc.item():.4f}")


if __name__ == "__main__":
    train()
