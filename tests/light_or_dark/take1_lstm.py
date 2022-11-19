# %%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,  LSTM,Activation
import numpy as np
from generator import generator
# %% Wall time: 20min 3s
model = Sequential()
model.add(LSTM(100, 
            activation='tanh', 
            recurrent_activation='hard_sigmoid'))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy', 
            optimizer='adam', 
            metrics=["binary_accuracy"])
output_shapes=([None,500,10], [None])
tgen = tf.data.Dataset.from_generator(lambda: generator(True,"./edf_files/lord/ex.h5",500,4,-20,label_func=lambda label: int(label == "dark")),output_types=(np.float32,np.float32), output_shapes=output_shapes)
vgen = tf.data.Dataset.from_generator(lambda: generator(False,"./edf_files/lord/ex.h5",500,4,-20,label_func=lambda label: int(label == "dark")),output_types=(np.float32,np.float32), output_shapes=output_shapes)
history = model.fit(tgen,
        epochs=100, 
        batch_size=4,
        validation_data= vgen)
#predict = model.predict(test, verbose=1)
# %%
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
def plot_history(h:Any,metrics:list[str]):
    loss = h['loss']
    val_loss = h['val_loss']
    #train_acc = h['binary_accuracy']
    #val_total_acc = h['val_binary_accuracy']
    epochs = range(len(loss))

    plt.figure(figsize=(12, 10))
    plt.rcParams['font.size'] = 25
    plt.plot(epochs, loss,linestyle = "solid" ,label = 'loss')
    plt.plot(epochs, val_loss,linestyle = "solid" , label= 'valid loss')
    for mn in metrics:
        acc = h[mn]
        val_acc = h['val_'+mn]
        plt.plot(epochs, acc, linestyle = "solid", label = mn)
        plt.plot(epochs, val_acc, linestyle = "solid", label= 'valid '+mn)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
    plt.grid()
    plt.show()
    plt.close()
plot_history(history.history,metrics=["binary_accuracy"])
def save_history(dir_path:str,h:Any):
    hdf = pd.DataFrame(history.history)
    if dir_path[-1] != "/":
        dir_path = dir_path + "/"
    hdf.to_csv(f"{dir_path}history.csv")
save_history(".",history.history)
# %%
