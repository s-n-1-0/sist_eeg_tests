# %%
from .base_generator import BaseGeneratorMaker
import random
from typing import Any, Callable
import h5py
import numpy as np
def transpose_raw1d(_:int,batch:np.ndarray):
    return batch.transpose([0,2,1])
def transpose_raw2d(_:int,batch:np.ndarray):
    return batch[:,:,:,None]#.transpose([0,2,1])
class RawGeneratorMaker(BaseGeneratorMaker):
    def __init__(self, hdf_path: str,istest=False,valid_keys = []) -> None:
        group_name = "std"
        super().__init__(hdf_path, group_name,istest=istest,vaild_keys = valid_keys)

    def make_generators(self, batch_size: int,pick_func: Callable[[h5py.Dataset, bool], np.ndarray]):
        return super().make_generators(batch_size,pick_func,transpose_func=transpose_raw1d)
    def make_2d_generators(self, batch_size: int,pick_func: Callable[[h5py.Dataset, bool], np.ndarray]):
        return super().make_generators(batch_size,pick_func,transpose_func=transpose_raw2d)
# %%
