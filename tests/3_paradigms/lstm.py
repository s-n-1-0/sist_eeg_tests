# %%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,  LSTM,Activation
import numpy as np
from generator import RawGeneratorMaker,dataset_dir_path
from summary import summary
# %% 
offset = 0
back = 500
ch_list = [12, 13, 14, 35, 36, 8, 7, 9, 10, 18, 17, 19, 20]
batch_size = 32
# %%
model = Sequential()
model.add(LSTM(64,return_sequences=True,dropout=0.5))
model.add(LSTM(32,dropout=0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["binary_accuracy"])
output_shapes=([None,back,len(ch_list)], [None])

def pick_func(signal:np.ndarray,mode:bool):
    return signal[()][ch_list,offset:back+offset]

maker = RawGeneratorMaker(f"{dataset_dir_path}/3pdataset.h5")
tgen,vgen = maker.make_generators(batch_size,pick_func=pick_func)
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
summary(model,history,vgen,f"./saves/3p/lstm_raw_{len(ch_list)}_A")
