# %%
import random
from typing import Any, Callable
import h5py
import numpy as np
rate = 0.6
from utils import signals_standardization
def make_generators(path:str,erp_size:int,batch_size:int,border:int,label_func:Callable[[str],Any],pick_func:Callable[[h5py.Dataset,bool],np.ndarray]):
    with h5py.File(path, 'r') as hf:
        group = hf["annotations/Marker"]
        origin_keys = list(group.keys())
        all_keys = origin_keys[:]
        random.shuffle(all_keys)
        train_all_keys  = all_keys[:border]
        train_dark_keys = []
        train_light_keys = []
        for key in train_all_keys:
            if group[key].attrs["label"] == "dark":
                train_dark_keys.append(key)
            else:
                train_light_keys.append(key)
    def make_train_generator():
        with h5py.File(path, 'r') as hf:
            group = hf["annotations/Marker"]
            def get_dataset(keys:list[str]):
                random.shuffle(keys)
                if random.random() >= rate:
                    erp = []
                    for i in range(erp_size):
                        dataset = group[keys[i]]
                        erp.append(pick_func(dataset,False))
                    return signals_standardization(np.array(erp).sum(axis=0) / erp_size)#std(std(np.array(erp).sum(axis=0) / erp_size) * 0.2 + pick_func(group[keys[erp_size + 1]],False)* 0.8)
                else:
                    return pick_func(group[keys[0]],False)
            itr_count = 0
            batch_count = 0
            steps = len(train_all_keys) // batch_size
            x = []
            y = []
            while itr_count < steps:
                batch_count += 1
                is_dark = random.random() >= 0.5
                x.append(get_dataset(train_dark_keys if is_dark else train_light_keys))
                y.append(label_func("dark" if is_dark else "light"))
                
                if batch_count % batch_size == 0:
                    yield (np.array(x,dtype=np.float32).transpose(0,2,1),np.array(y,dtype=np.float32))
                    x = []
                    y = []
                    itr_count += 1 
    def make_valid_generator():
        with h5py.File(path, 'r') as hf:
            group = hf["annotations/Marker"]
            keys = all_keys[border:]
            count = 0
            x = []
            y = []
            while count < len(keys):
                dataset = group[keys[count]]
                count += 1
                x.append(pick_func(dataset,False))
                y.append(label_func(dataset.attrs["label"]))
                if count % batch_size == 0:
                    yield (np.array(x,dtype=np.float32).transpose(0,2,1),np.array(y,dtype=np.float32))
                    x = []
                    y = []

    return make_train_generator,make_valid_generator

def make_test_generator(**kwargs):
    kwargs["border"] = 0
    return make_generators(**kwargs)[1]

# %%
