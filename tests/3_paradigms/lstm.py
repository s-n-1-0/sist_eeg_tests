# %%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,  LSTM,Activation
import numpy as np
from generator import RawGeneratorMaker,dataset_dir_path
from pickfunc import RawPickFuncMaker
from summary import summary
# %% 
offset = 500
back = 500
pfm = RawPickFuncMaker(back,2000)
batch_size = 32
# %%
model = Sequential()
model.add(LSTM(16,dropout=0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["binary_accuracy"])
output_shapes=([None,back,len(pfm.ch_list)], [None])

def pick_func(signal:np.ndarray,_:bool):
    return signal[()][pfm.ch_list,offset:back+offset]

maker = RawGeneratorMaker(f"{dataset_dir_path}/3pdataset.h5")
tgen,vgen = maker.make_generators(batch_size,pick_func=pfm.make_pick_func(offset))
def from_generator(gen):
    return tf.data.Dataset.from_generator(gen,output_types=(np.float32,np.float32), output_shapes=output_shapes)
tgen = from_generator(tgen)
vgen = from_generator(vgen)
history = model.fit(tgen,
        epochs=50, 
        batch_size=batch_size,
        validation_data= vgen)
#predict = model.predict(test, verbose=1)
# %%
summary(model,history,vgen,f"./saves/3p/lstm_raw_{len(pfm.ch_list)}_{maker.split_mode}")
