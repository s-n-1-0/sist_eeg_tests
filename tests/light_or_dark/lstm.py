# %%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,  LSTM,Activation
import numpy as np
from generator import generator
# %%
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
history
# %%
