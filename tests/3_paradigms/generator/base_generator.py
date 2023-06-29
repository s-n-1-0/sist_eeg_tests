# %%
import random
from typing import Any, Callable
import h5py
import numpy as np

class BaseGeneratorMaker():
    def __init__(self,hdf_path:str,group_name:str) -> None:
        self.hdf_path = hdf_path
        with h5py.File(hdf_path, 'r') as hf:
            group = hf["prepro/"+group_name]
            origin_keys = list(group.keys())
            #random.shuffle(shuffled_keys)
            train_keys = []
            vaild_keys = []
            for sk in origin_keys[:]:
                subject = group[sk].attrs["subject"]
                if subject < 50:
                    train_keys.append(sk)
                else:
                    vaild_keys.append(sk)
            self.group_name = group_name
            self.origin_keys = origin_keys
            print(len(train_keys),len(vaild_keys))
            self.train_keys = train_keys
            self.valid_keys = vaild_keys
    def make_generators(
            self,
            batch_size:int,
            pick_func:Callable[[h5py.Dataset,bool],np.ndarray],
            transpose_func:Callable[[np.ndarray],np.ndarray]
            ):
        def make_generator(mode:bool):
            def generator():
                with h5py.File(self.hdf_path, 'r') as hf:
                    group = hf["prepro/" + self.group_name]
                    if mode:
                        keys  = self.train_keys[:]
                        random.shuffle(keys)
                    else:
                        keys = self.valid_keys[:]
                    count = 0
                    x = []
                    y = []
                    while count < len(keys):
                        dataset = group[keys[count]]
                        count += 1
                        x.append(pick_func(dataset,mode))
                        y.append(int(dataset.attrs["label"] == "right"))
                        if count % batch_size == 0:
                            yield (transpose_func(np.array(x,dtype=np.float32)),np.array(y,dtype=np.float32))
                            x = []
                            y = []

            return generator
        return make_generator(True),make_generator(False)
# %%