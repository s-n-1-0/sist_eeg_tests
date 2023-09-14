import h5py
import random
import numpy as np
class BasePickFuncMaker():
    def __init__(self) -> None:
        self.ch_list = list(range(13))
        #self.ch_list = [12, 13, 14]
class RawPickFuncMaker(BasePickFuncMaker):
    def __init__(self,sample_size:int,max_sample_size) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.max_sample_size = max_sample_size
    def make_pick_func(self,offset = 0,is_random_valid:bool = False):
        def pick_func(signal: h5py.Dataset, is_train: bool):
            if not is_train and is_random_valid:
                random_offset = random.randint(0,self.max_sample_size-self.sample_size)
                return [signal[()][self.ch_list,random_offset:self.sample_size+random_offset]]
            return [signal[()][self.ch_list,offset:self.sample_size+offset]]
        return pick_func
    
    #ランダムの時点を開始地点としてサンプルをとる
    def make_random_pick_func(self):
        def pick_func(signal: h5py.Dataset, _: bool):
            random_offset = random.randint(0,self.max_sample_size-self.sample_size)
            return [signal[()][self.ch_list,random_offset:self.sample_size+random_offset]]
        return pick_func
class MultiRawPickFuncMaker(BasePickFuncMaker):
    def __init__(self,hand_sample_size:int,rest_path:str) -> None:
        super().__init__()
        self.hand_sample_size = hand_sample_size
        self.rest_h5 = h5py.File(rest_path)
    def make_random_pick_func(self,max_sample_size:int):
        group = self.rest_h5["rest/3p"]
        def pick_func(signal: h5py.Dataset, _: bool):
            session = signal.attrs["session"]
            subject = signal.attrs["subject"]
            random_offset = random.randint(0,max_sample_size-self.hand_sample_size)
            return [signal[()][self.ch_list,random_offset:self.hand_sample_size+random_offset],group[f"{session}/{subject}"]]
        return pick_func
class PsdPickFuncMaker(BasePickFuncMaker):
    def __init__(self) -> None:
        super().__init__()
        self.psd_size = 13
    def make_pick_func(self):
        def pick_func(signal:h5py.Dataset,_:bool):
            return [signal[()][self.ch_list,8:21]]
        return pick_func
class DwtPickFuncMaker(BasePickFuncMaker):
    def __init__(self) -> None:
        super().__init__()
    def make_pick_func(self):
        def pick_func(dataset:h5py.Dataset,_:bool):
            signal = dataset[()][self.ch_list,:,:]
            s1 = signal[:,0,:65] #65dwtの元のサイズ
            s2 = signal[:,1,:130]
            signal = np.concatenate([s1,s2],axis=1)
            return [signal]
        return pick_func
