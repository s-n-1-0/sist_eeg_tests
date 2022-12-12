# %%
import tensorflow as tf
from keras import Model
from keras.models import Sequential
from keras.layers import Dense,  Flatten,MaxPooling1D,Dropout,Conv1D,LeakyReLU,concatenate
import numpy as np
from erp_generator import make_generators
from utils.history import save_history,plot_history
from utils.keras.layers import Maxout
# %% 
back = 300
ch = 10
batch_size = 4
# %%
class STF7(Model):
    def __init__(self):
        super(STF7,self).__init__()
        self.conv1d_1 = Conv1D(
            filters=6,
            kernel_size= 5,
            padding='same'
        )
        #↓ここは高速化できるらしい...(参考モデルのnet_stf7.py参照)
        self.conv1d_2 = Conv1D(
            filters=6,
            kernel_size= 16,
            strides=16
        )
        self.conv1d_3 = Conv1D(
            filters=32,
            kernel_size= 7,
            padding='same',
            activation=LeakyReLU()
        )
        self.pooling1d_1 = MaxPooling1D(
            pool_size=3,
            strides=2,
            padding="same"
        )
        self.conv1d_4 = Conv1D(
            filters=64,
            kernel_size= 5,
            padding='same'
        )
        self.maxout_1 = Maxout(32)
        self.pooling1d_2 = MaxPooling1D(
            pool_size=12,
            strides=8,
            padding="same"
        )
        self.l1 = Dense(512*4)
        self.l2 = Dense(512*4)
        self.maxout_2 = Maxout(512)
        self.l3 = Dense(1,activation='sigmoid')

        self.flatten = Flatten()
        self.dropout = Dropout(0.5)
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
        x = concatenate([x1,x2],axis=-1)
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
model = STF7()
model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.00001), #0.000001
            metrics=["binary_accuracy"])
output_shapes=([None,back,ch], [None])

def take6_pick(signal:np.ndarray,mode:bool):
    return signal[:,50:50 + back]

tgen,vgen = make_generators("./dataset/lord2/ex.h5",batch_size,-216,label_func=lambda label: int(label == "dark"),pick_func=take6_pick)
def from_generator(gen):
    return tf.data.Dataset.from_generator(gen,output_types=(np.float32,np.float32), output_shapes=output_shapes)
tgen = from_generator(tgen)
vgen = from_generator(vgen)

# %%
history = model.fit(tgen,
        epochs=20, 
        batch_size=batch_size,
        validation_data= vgen)
#predict = model.predict(test, verbose=1)
# %%
plot_history(history.history,metrics=["binary_accuracy"])
plot_history(history.history,metrics=["binary_accuracy"],is_loss=False)
save_history(".",history.history)

# %%
model.save(".\model_e500.h5",save_format="h5")
# %%
history = model.fit(tgen,
        initial_epoch=20,
        epochs=40, 
        batch_size=batch_size,
        validation_data= vgen)