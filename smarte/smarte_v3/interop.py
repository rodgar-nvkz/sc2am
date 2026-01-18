import ctypes
from multiprocessing import RawArray, RawValue

import numpy as np
import torch
from torch import nn


class SharedWeights:
    """Shared memory for model weights with version tracking"""

    def __init__(self, model: nn.Module):
        flat_params = nn.utils.parameters_to_vector(model.parameters())
        self._num_params = flat_params.numel()
        self.shared_array = RawArray(ctypes.c_float, self._num_params)
        self.version = RawValue(ctypes.c_int, 0)
        self.push(model)

    def push(self, model: nn.Module):
        """Copy model weights to shared memory."""
        params = nn.utils.parameters_to_vector(model.parameters())
        np_array = np.frombuffer(self.shared_array, dtype=np.float32)
        np.copyto(np_array, params.detach().cpu().numpy())
        # Atomic increment (no lock needed for single writer)
        self.version.value += 1

    def pull(self, model: nn.Module) -> int:
        """Copy weights from shared memory to model. Returns version"""
        # Copy to avoid race conditions during parameter update
        np_array = np.frombuffer(self.shared_array, dtype=np.float32).copy()
        nn.utils.vector_to_parameters(torch.from_numpy(np_array), model.parameters())
        return self.version.value

    def get_version(self) -> int:
        """Lock-free version read. May be slightly stale, which is acceptable."""
        return self.version.value
