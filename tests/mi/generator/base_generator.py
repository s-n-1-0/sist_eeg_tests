# %%
import random
from typing import Any, Callable,Union
import h5py
import numpy as np

class BaseGeneratorMaker():
    def __init__(self,hdf_path:str,
                 group_name:str,
                 istest:bool = False,
                 vaild_keys = []) -> None:
        self.hdf_path = hdf_path
        with h5py.File(hdf_path, 'r') as hf:
            group = hf["prepro/"+group_name]
            origin_keys = list(group.keys())
            #random.shuffle(shuffled_keys)
            train_keys = []
            if istest:
                vaild_keys = origin_keys[:]
            else:
                if len(vaild_keys) > 0:
                    print("検証データ指定モード")
                    for sk in origin_keys[:]:
                        dataset = group[sk].attrs["dataset"]
                        if sk not in vaild_keys:
                            train_keys.append(sk)
                    self.split_mode = "B"
                elif False:
                    print("被験者別モード")
                    subject_list = np.random.randint(1,55,size=5) #[54,34,21,35,1]
                    print("検証被験者:" + str(subject_list))
                    for sk in origin_keys[:]:
                        subject = group[sk].attrs["subject"]
                        if subject not in subject_list:
                            train_keys.append(sk)
                        else:
                            vaild_keys.append(sk)
                    self.split_mode = "A"
                    self.subject_list = subject_list
                else:
                    print("一部セッション別モード")
                    shuffled_keys = origin_keys[:]
                    random.shuffle(shuffled_keys)
                    for sk in shuffled_keys:
                        session = group[sk].attrs["session"]
                        if session == 2 and len(vaild_keys) < 700:
                            vaild_keys.append(sk)
                        else:
                            train_keys.append(sk)
                    self.split_mode = "B"
            self.group_name = group_name
            self.origin_keys = origin_keys
            print(len(train_keys),len(vaild_keys))
            self.train_keys = train_keys
            self.valid_keys = vaild_keys
    def make_generators(
            self,
            batch_size:int,
            pick_func:Callable[[h5py.Dataset,bool],np.ndarray],
            transpose_func:Callable[[np.ndarray],np.ndarray],
            label_func:Callable[[h5py.Dataset],Union[int,np.ndarray]] = lambda dataset:int(dataset.attrs["label"] == "right")
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
                        x_list = pick_func(dataset,mode)
                        if len(x) == 0:
                            x = [[xx] for xx in x_list]
                        else:
                            for i,xx in enumerate(x_list):
                                x[i].append(xx)
                        y.append(int(label_func(dataset)))
                        if count % batch_size == 0:
                            ret_x = [transpose_func(i,np.array(x[i],dtype=np.float32)) for i in range(len(x))]
                            if len(ret_x) == 1:
                                yield (ret_x[0],np.array(y,dtype=np.float32))
                            else:
                                yield ({"input_1": ret_x[0], "input_2": ret_x[1]},np.array(y,dtype=np.float32))
                            x = []
                            y = []

            return generator
        return make_generator(True),make_generator(False)
# %%
