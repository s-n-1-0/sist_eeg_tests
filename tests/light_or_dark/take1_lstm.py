# %%
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,  LSTM,Activation
import numpy as np
from generator import generator
from utils.history import save_history,plot_history
# %% 
# take1 Wall time: 20min 3s
# take2,3 Wall time: 7min 14s
back = 500 #take 2,3 = 160 # take1,4 = 500
ch = 2
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
def make_generator(is_train:bool):
    def take1_pick(signal:np.ndarray):
        r = random.randint(0,25) # ランダム要素
        return signal[:,r: r + back]
    def take2_pick(signal:np.ndarray):
        return signal[[0,5],:back]
    return tf.data.Dataset.from_generator(lambda: generator(is_train,"./edf_files/lord/ex.h5",4,-20,label_func=lambda label: int(label == "dark"),pick_func=take2_pick),output_types=(np.float32,np.float32), output_shapes=output_shapes)
tgen = make_generator(True)
vgen = make_generator(False)
history = model.fit(tgen,
        epochs=500, 
        batch_size=4,
        validation_data= vgen)
#predict = model.predict(test, verbose=1)
# %%
plot_history(history.history,metrics=["binary_accuracy"])
plot_history(history.history,metrics=[])
save_history(".",history.history)

# %%
model.save(".\model_e500.h5",save_format="h5")
# %%
