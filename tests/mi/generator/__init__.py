from .raw_generator import RawGeneratorMaker
from .psd_generator import PsdGeneratorMaker
from .dwt_generator import DwtGeneratorMaker
import numpy as np
dataset_dir_path = "//172.16.88.200/private/2221012"

#LDAやSVM用
def merge_gen(gen,init_shape):
    xd = np.zeros(init_shape)
    yd = []
    for x,y in gen():
        for i in range(x.shape[0]):
            _x = x[i,:,:].T.reshape(-1,1).T
            xd = np.vstack([xd,_x])
            yd.append(y[i])
    yd = np.array(yd)
    return xd,yd