import tensorflow as tf
import numpy as np
back = 500
ch = 10
batch_size = 32
dataset_path = "./dataset/lord2/train/ex.h5"
output_shapes=([None,back,ch], [None])

def from_generator(gen):
    return tf.data.Dataset.from_generator(gen,output_types=(np.float32,np.float32), output_shapes=output_shapes)

def pick(signal:np.ndarray,mode:bool):
    return signal[:,:back]