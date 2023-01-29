# %%
import random
from typing import Any, Callable
import h5py
import numpy as np
def make_generators(path:str,
                    batch_size:int,
                    dataset_size:tuple[int], #ラベル1種ごとのサイズ
                    border:int,
                    pick_func:Callable[[h5py.Dataset,bool],np.ndarray],
                    transpose_rule = [0,2,1]):
    with h5py.File(path, 'r') as hf:
        marker_group = hf["annotations/Marker"]
        wait_group = hf["annotations/Wait"]
        origin_marker_keys = ["Marker/" + k for k in list(marker_group.keys())]
        origin_wait_keys = ["Wait/" + k for k in list(wait_group.keys())]
        marker_all_keys = origin_marker_keys[:]
        random.shuffle(marker_all_keys)
        wait_all_keys = origin_wait_keys[:]
        random.shuffle(wait_all_keys)
        all_keys = marker_all_keys[:dataset_size*2] + wait_all_keys[:dataset_size]
        random.shuffle(all_keys)
    def label_func(key:str,label:str):
        if key.startswith("Marker"):
            return [0,1,0] if label == "light" else [0,0,1]
        else:
            return [1,0,0]
    def make_generator(mode:bool):
        def generator():
            with h5py.File(path, 'r') as hf:
                group = hf["annotations"]
                if mode:
                    keys  = all_keys[:border]
                    random.shuffle(keys)
                else:
                    keys = all_keys[border:]
                count = 0
                x = []
                y = []
                while count < len(keys):
                    key = keys[count]
                    dataset = group[key]
                    count += 1
                    x.append(pick_func(dataset,mode))
                    y.append(label_func(key,dataset.attrs["label"]))
                    if count % batch_size == 0:
                        yield (np.array(x,dtype=np.float32).transpose(*transpose_rule),np.array(y,dtype=np.float32))
                        x = []
                        y = []

        return generator
    return make_generator(True),make_generator(False)
def make_test_generator(**kwargs):
    kwargs["border"] = 0
    return make_generators(**kwargs)[1]
# %%
