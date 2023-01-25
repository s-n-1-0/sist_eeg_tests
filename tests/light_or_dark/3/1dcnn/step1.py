# %%
import sys
import os
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense,Activation,Dropout,Conv1D,MaxPooling1D,Flatten,BatchNormalization
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from generator import make_generators,split_dataset
from common import *
from utils.history import save_history,plot_history
from sklearn.metrics import confusion_matrix
import pandas as pd
# %% split dataset
step1_dataset,step2_dataset = split_dataset(dataset_path,800)
# %%
model = Sequential()
model.add(Conv1D(
            filters=32,
            kernel_size= 50,
            strides=5
        ))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Conv1D(
            filters=64,
            kernel_size= 25,
            strides=3
        ))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Conv1D(
            filters=64,
            kernel_size= 7,
            padding='same'
        ))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(MaxPooling1D(
            pool_size=3,
            strides=2,
            padding="same"
        ))
model2 = Sequential()
model2.add(model)
model2.add(Flatten())
model2.add(Dense(128,activation="sigmoid"))
model2.add(Dropout(0.4))
model2.add(Dense(128,activation="sigmoid"))
model2.add(Dropout(0.4))
model2.add(Dense(1,activation="sigmoid"))
model2.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["binary_accuracy"])

tgen,vgen = make_generators(True,dataset_path,step1_dataset,batch_size,pick_func=pick)
tgen = from_generator(tgen)
vgen = from_generator(vgen)
model2.build(output_shapes[0])
model2.summary()
# %%
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        min_lr=0.00001
                )
history = model2.fit(tgen,
        epochs=200, 
        batch_size=batch_size,
        validation_data= vgen,
        callbacks=[reduce_lr])
# %%
plot_history(history.history,metrics=["binary_accuracy"])
plot_history(history.history,metrics=["binary_accuracy"],is_loss=False)
save_history(".",history.history)

# %%
def save_dataset_list(fn:str,data:list):
    with open(fn, 'w') as f:
        for x in data:
            f.write(x + "\n")
save_dataset_list("./step1_1.txt",step1_dataset[1])
save_dataset_list("./step2_0.txt",step2_dataset[0])
save_dataset_list("./step2_1.txt",step2_dataset[1])
model.save("./step1_model.h5")
model2.save("./step1_model2.h5")
# %% predict
labels = [1, 0]
_y_pred = model2.predict(vgen, verbose=1)
y_pred = [1.0 if p[0] > 0.5 else 0 for p in _y_pred]
x_valid = []
y_true = []
for v in vgen:
    x_valid += list(v[0].numpy())
    y_true +=list(v[1].numpy())
cm = confusion_matrix(y_true, y_pred, labels=labels)
columns_labels = ["pred_" + str(l) for l in labels]
index_labels = ["true_" + str(l) for l in labels]
cm = pd.DataFrame(cm,columns=columns_labels, index=index_labels)
print(cm.to_markdown())
ans_r = [c == a  for c,a in zip(y_pred,y_true)]
print(ans_r.count(True)/len(ans_r))
# %%
