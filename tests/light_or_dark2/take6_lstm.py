# %%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,  LSTM,Activation,Dropout
import numpy as np
from generators.erp_generator import make_generators
from utils.history import save_history,plot_history
# %% 
back = 100
ch = 10
batch_size = 4
# %%
model = Sequential()
model.add(LSTM(64,return_sequences=True,dropout=0.5))
model.add(LSTM(32,dropout=0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.00001), #0.000001
            metrics=["binary_accuracy"])
output_shapes=([None,back,ch], [None])

def take6_pick(signal:np.ndarray,mode:bool):
    return signal[:,50:50 + back]

tgen,vgen = make_generators("./dataset/lord2/ex.h5",160,batch_size,-162,label_func=lambda label: int(label == "dark"),pick_func=take6_pick)
def from_generator(gen):
    return tf.data.Dataset.from_generator(gen,output_types=(np.float32,np.float32), output_shapes=output_shapes)
tgen = from_generator(tgen)
vgen = from_generator(vgen)
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
