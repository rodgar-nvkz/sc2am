import os

os.environ.setdefault("RL_WARNINGS", "0")  # Tensordict with CudaGraphs
os.environ.setdefault("EXCLUDE_TD_FROM_PYTREE", "1")  # Tensordict with CudaGraphs
