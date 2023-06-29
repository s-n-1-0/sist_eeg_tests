# %%
from .base_generator import BaseGeneratorMaker
import random
from typing import Any, Callable
import h5py
import numpy as np

class RawGeneratorMaker(BaseGeneratorMaker):
    def __init__(self, hdf_path: str) -> None:
        group_name = "std"
        super().__init__(hdf_path, group_name)
    def make_generators(self, batch_size: int,pick_func: Callable[[h5py.Dataset, bool], np.ndarray]):
        def transpose_func(batch:np.ndarray):
             return batch.transpose([0,2,1])
        return super().make_generators(batch_size,pick_func,transpose_func=transpose_func)
# %%
