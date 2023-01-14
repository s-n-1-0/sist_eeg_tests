# %%
import random
from typing import Any, Callable
import h5py
import numpy as np

def split_dataset(path:str,train_size):
    with h5py.File(path, 'r') as hf:
            marker_group = hf["annotations/Marker"]
            wait_group = hf["annotations/Wait"]
            origin_marker_keys = ["Marker/" + k for k in list(marker_group.keys())]
            origin_wait_keys = ["Wait/" + k for k in list(wait_group.keys())]
            marker_all_keys = origin_marker_keys[:]
            random.shuffle(marker_all_keys)
            wait_all_keys = origin_wait_keys[:]
            random.shuffle(wait_all_keys)
            step1_train = marker_all_keys[:train_size] + wait_all_keys[:train_size]
            step1_valid = marker_all_keys[train_size:] + wait_all_keys[train_size:]
            step2_train = marker_all_keys[:train_size]
            step2_valid = marker_all_keys[train_size:] 
    return (step1_train,step1_valid),(step2_train,step2_valid)
            
def make_generators(is_step1:bool,
                    path:str,
                    dataset:tuple[list],
                    batch_size:int,
                    pick_func:Callable[[h5py.Dataset,bool],np.ndarray],
                    transpose_rule = [0,2,1]):
    train,valid = dataset
    def make_generator(mode:bool):
        def generator():
            with h5py.File(path, 'r') as hf:
                group = hf["annotations"]
                if mode:
                    keys  = train[:]
                    random.shuffle(keys)
                else:
                    keys = valid
                count = 0
                x = []
                y = []
                while count < len(keys):
                    _dataset = group[keys[count]]
                    x.append(pick_func(_dataset,mode))
                    if is_step1:
                        y.append(int(keys[count].startswith("Marker")))
                    else:
                        y.append(int(_dataset.attrs["label"] == "dark"))
                    count += 1
                    if count % batch_size == 0:
                        yield (np.array(x,dtype=np.float32).transpose(*transpose_rule),np.array(y,dtype=np.float32))
                        x = []
                        y = []

        return generator
    return make_generator(True),make_generator(False)
# %%
