# %%
import random
from typing import Any, Callable
import h5py
import numpy as np
def make_generators(path:str,batch_size:int,border:int,label_func:Callable[[str],Any],pick_func:Callable[[np.ndarray],np.ndarray]):
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
                steps = len(keys) // batch_size
                count = 0
                while count < steps:
                    x = []
                    y = []
                    for i in range(batch_size):
                        dataset = group[keys[i + count * batch_size]]
                        x.append(pick_func(dataset))
                        y.append(label_func(dataset.attrs["label"]))
                    yield (np.array(x,dtype=np.float32).transpose(0,2,1),np.array(y,dtype=np.float32))
                    count += 1
        return generator
    return make_generator(True),make_generator(False)
# %%
