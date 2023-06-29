# %%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,  LSTM,Activation
import numpy as np
from generator import RawGeneratorMaker
from summary import summary
root_path = "//172.16.88.200/private/2221012/MIOnly_FTP_EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms"
# %% 
offset = 0
back = 500
ch = 3
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
output_shapes=([None,back,ch], [None])

def pick_func(signal:np.ndarray,mode:bool):
    return signal[12:15,offset:back+offset]

maker = RawGeneratorMaker(f"{root_path}/3pdataset.h5")
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
summary(model,history,vgen)
