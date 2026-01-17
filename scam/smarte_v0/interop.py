import ctypes
from multiprocessing import RawArray, Value

import numpy as np
from torch import from_numpy, nn


class SharedWeights:
    """Shared memory for model weights with version tracking."""

    def __init__(self, model: nn.Module):
        flat_params = nn.utils.parameters_to_vector(model.parameters())
        self.shared_array = RawArray(ctypes.c_float, flat_params.numel())  # /shm array
        self.version = Value(ctypes.c_int, 0)  # Version tracker
        self.push(model)

    def push(self, model: nn.Module):
        """Copy model weights to shared memory."""
        params = nn.utils.parameters_to_vector(model.parameters())
        np_array = np.frombuffer(self.shared_array, dtype=np.float32)
        np.copyto(np_array, params.detach().cpu().numpy())

        with self.version.get_lock():
            self.version.value += 1

    def pull(self, model: nn.Module) -> int:
        """Copy weights from shared memory to model. Returns version."""
        np_array = np.frombuffer(self.shared_array, dtype=np.float32).copy()
        nn.utils.vector_to_parameters(from_numpy(np_array), model.parameters())

        with self.version.get_lock():
            return self.version.value

    def get_version(self) -> int:
        with self.version.get_lock():
            return self.version.value
