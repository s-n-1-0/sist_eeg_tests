# %%
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,  LSTM,Activation
import numpy as np
from generator import make_generators
from utils.history import save_history,plot_history
# %% 
# take1 Wall time: 20min 3s
# take2,3 Wall time: 7min 14s
back = 500 #take 2,3 = 160 # take1,4 = 500
ch = 10
# %%
model = Sequential()
model.add(LSTM(100, 
            activation='tanh', 
            recurrent_activation='sigmoid'))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy', 
            optimizer='adam', 
            metrics=["binary_accuracy"])
output_shapes=([None,back,ch], [None])

def take6_pick(signal:np.ndarray,mode:bool):
    return signal[:,:back]

tgen,vgen = make_generators("./dataset/lord2/ex.h5",4,-20,label_func=lambda label: int(label == "dark"),pick_func=take6_pick)
def from_generator(gen):
    return tf.data.Dataset.from_generator(gen,output_types=(np.float32,np.float32), output_shapes=output_shapes)
tgen = from_generator(tgen)
vgen = from_generator(vgen)
history = model.fit(tgen,
        epochs=500, 
        batch_size=4,
        validation_data= vgen)
#predict = model.predict(test, verbose=1)
# %%
plot_history(history.history,metrics=["binary_accuracy"])
plot_history(history.history,metrics=["binary_accuracy"],is_loss=False)
save_history(".",history.history)

# %%
model.save(".\model_e500.h5",save_format="h5")
# %%
