# %%
import random
from typing import Any, Callable
import h5py
import numpy as np
def make_generators(path:str,
                    batch_size:int,
                    border:int,
                    label_func:Callable[[str],Any],
                    pick_func:Callable[[h5py.Dataset,bool],np.ndarray],
                    transpose_rule = [0,2,1]):
    with h5py.File(path, 'r') as hf:
        group = hf["annotations/Marker"]
        origin_keys = list(group.keys())
        all_keys = origin_keys[:]
        random.shuffle(all_keys)
    def make_generator(mode:bool):
        def generator():
            with h5py.File(path, 'r') as hf:
                group = hf["annotations/Marker"]
                if mode:
                    keys  = all_keys[:border]
                    random.shuffle(keys)
                else:
                    keys = all_keys[border:]
                count = 0
                x = []
                y = []
                while count < len(keys):
                    dataset = group[keys[count]]
                    count += 1
                    x.append(pick_func(dataset,mode))
                    y.append(label_func(dataset.attrs["label"]))
                    if count % batch_size == 0:
                        yield (np.array(x,dtype=np.float32).transpose(*transpose_rule),np.array(y,dtype=np.float32))
                        x = []
                        y = []

        return generator
    return make_generator(True),make_generator(False)
# %%
