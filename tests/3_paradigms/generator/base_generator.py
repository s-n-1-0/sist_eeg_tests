# %%
import random
from typing import Any, Callable
import h5py
import numpy as np

class BaseGeneratorMaker():
    def __init__(self,hdf_path:str,group_name:str) -> None:
        self.hdf_path = hdf_path
        with h5py.File(hdf_path, 'r') as hf:
            group = hf[group_name]
            origin_keys = list(group.keys())
            shuffled_keys = origin_keys[:]
            random.shuffle(shuffled_keys)
            self.group_name = group_name
            self.origin_keys = origin_keys
            self.shuffled_keys = shuffled_keys
    def make_generators(
            self,
            batch_size:int,
            valid_border:int,
            pick_func:Callable[[h5py.Dataset,bool],np.ndarray],
            transpose_func:Callable[[np.ndarray],np.ndarray]
            ):
        def make_generator(mode:bool):
            shuffled_keys = self.shuffled_keys
            def generator():
                with h5py.File(self.hdf_path, 'r') as hf:
                    group = hf[self.group_name]
                    if mode:
                        keys  = shuffled_keys[:valid_border]
                        random.shuffle(keys)
                    else:
                        keys = shuffled_keys[valid_border:]
                    count = 0
                    x = []
                    y = []
                    while count < len(keys):
                        dataset = group[keys[count]]
                        count += 1
                        x.append(pick_func(dataset,mode))
                        #TODO: leftを1とするかrightを1とするか
                        y.append(int(dataset.attrs["label"] == "right"))
                        if count % batch_size == 0:
                            yield (transpose_func(np.array(x,dtype=np.float32)),np.array(y,dtype=np.float32))
                            x = []
                            y = []

            return generator
        return make_generator(True),make_generator(False)
# %%
