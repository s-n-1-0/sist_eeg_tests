import h5py
import random
class BasePickFuncMaker():
    def __init__(self) -> None:
        self.ch_list = [12, 13, 14, 35, 36, 8, 7, 9, 10, 18, 17, 19, 20]

class RawPickFuncMaker(BasePickFuncMaker):
    def __init__(self,sample_size:int) -> None:
        super().__init__()
        self.sample_size = sample_size
    def make_pick_func(self,offset = 0):
        def pick_func(signal: h5py.Dataset, _: bool):
            return [signal[()][self.ch_list,offset:self.sample_size+offset]]
        return pick_func
    
    #ランダムの時点を開始地点としてサンプルをとる
    def make_random_pick_func(self,max_sample_size:int):
        def pick_func(signal: h5py.Dataset, _: bool):
            random_offset = random.randint(0,max_sample_size-self.sample_size)
            return [signal[()][self.ch_list,random_offset:self.sample_size+random_offset]]
        return pick_func
        
class PsdPickFuncMaker(BasePickFuncMaker):
    def __init__(self) -> None:
        super().__init__()
        self.psd_size = 50
    def make_pick_func(self):
        def pick_func(signal:h5py.Dataset,_:bool):
            return [signal[()][self.ch_list,31:81]]
        return pick_func
