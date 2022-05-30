# %%
import json
import numpy as np
import tensorflow as tf

from keras import Model,callbacks,optimizers,layers,backend as K
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import clip_ops,math_ops
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),"tests/eeg_dataset_cnn/src"))
from run_modules import Maxout,EEGDataset,total_acc,binary_acc,recall,precision
# %% データセットの読み込み
dataset = EEGDataset()
dataset.read_dataset()

# %% model class
class STF7(Model):
    def __init__(self):
        super(STF7,self).__init__()
        self.conv1d_1 = layers.Conv1D(
            filters=6,
            kernel_size= 5,
            padding='same',
            input_shape=(None,4096,32)
        )
        #↓ここは高速化できるらしい...(参考モデルのnet_stf7.py参照)
        self.conv1d_2 = layers.Conv1D(
            filters=6,
            kernel_size= 16,
            strides=16,
            input_shape=(4096,6)
        )
        self.conv1d_3 = layers.Conv1D(
            filters=32,
            kernel_size= 7,
            padding='same',
            activation=layers.LeakyReLU(),
            input_shape=(256,6)
        )
        self.pooling1d_1 = layers.MaxPooling1D(
            pool_size=3,
            strides=2,
            padding="same"
        )
        self.conv1d_4 = layers.Conv1D(
            filters=64,
            kernel_size= 5,
            padding='same',
            input_shape=(128,32)
        )
        self.maxout_1 = Maxout(32)
        self.pooling1d_2 = layers.MaxPooling1D(
            pool_size=12,
            strides=8,
            padding="same"
        )
        self.l1 = layers.Dense(512*4)
        self.l2 = layers.Dense(512*4)
        self.maxout_2 = Maxout(512)
        self.l3 = layers.Dense(6,activation='sigmoid')

        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
    @tf.function
    def call(self, x, training:bool=None):
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        x = self.pooling1d_1(x)
        x = self.conv1d_4(x)
        z = self.maxout_1(x)
        x1 = self.pooling1d_2(z[:,:-8,:])
        x1 = self.flatten(x1)
        x2 = z[:,-8:,:]
        x2 = self.flatten(x2)
        x = layers.concatenate([x1,x2],axis=-1)
        if training:
            x = self.dropout(x)
        x = self.l1(x)
        x = self.maxout_2(x)
        if training:
            x = self.dropout(x)
        x = self.l2(x)
        x = self.maxout_2(x)
        if training:
            x = self.dropout(x)
        x = self.l3(x)
        return x
# %%
epochs = 40
signal_size = 4096
batch_size = 32
model = STF7()
#ers = callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)
opt = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=[total_acc,binary_acc,recall,precision])
output_shapes=([None,signal_size,32], [None,6])
tgen = tf.data.Dataset.from_generator(lambda: dataset.make_train_generator(signal_size,batch_size), output_types=(np.float32, np.float32),output_shapes=output_shapes)
vgen = tf.data.Dataset.from_generator(lambda: dataset.make_valid_generator(signal_size,batch_size), output_types=(np.float32, np.float32),output_shapes=output_shapes)
history = model.fit(tgen,
            batch_size=batch_size,
            epochs=epochs,
            validation_data= vgen,
            )
model.summary()

# %%
