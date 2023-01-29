# %%
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense,Activation,Dropout,Conv1D,MaxPooling1D,Flatten,BatchNormalization
import numpy as np
from test1_generator import make_generators,split_dataset
from common import *
from utils.history import save_history,plot_history
from sklearn.metrics import confusion_matrix
import pandas as pd
# %% split dataset
step1_dataset,step2_dataset = split_dataset(dataset_path,800)
# %%
def make_model():
    model = Sequential()
    model.add(Conv1D(
                filters=64,
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
    model.add(Flatten())
    model.add(Dense(128,activation="sigmoid"))
    model.add(Dropout(0.4))
    model.add(Dense(1,activation="sigmoid"))
    return model
model = make_model()
model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["binary_accuracy"])

tgen,vgen = make_generators(True,dataset_path,step1_dataset,batch_size,pick_func=pick)
tgen = from_generator(tgen)
vgen = from_generator(vgen)
model.build(output_shapes[0])
model.summary()
# %%
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        min_lr=0.00001
                )
history = model.fit(tgen,
        epochs=40, 
        batch_size=batch_size,
        validation_data= vgen,
        callbacks=[reduce_lr])
# %%
plot_history(history.history,metrics=["binary_accuracy"])
plot_history(history.history,metrics=["binary_accuracy"],is_loss=False)
save_history(".",history.history)

# %% predict
_,valid_dataset = step1_dataset
valid_dataset = np.array(valid_dataset)
labels = [1, 0]
_y_pred = model.predict(vgen, verbose=1)
y_pred = [1 if p[0] > 0.5 else 0 for p in _y_pred]
y_true = []
for v in vgen:
    y_true +=list(v[1].numpy())
    
tp_flags = [t == p for t,p in zip(y_true, y_pred)]
tp_flags = tp_flags + [False] * (len(valid_dataset) - len(tp_flags))
good_valid_dataset = valid_dataset[tp_flags]
cm = confusion_matrix(y_true, y_pred, labels=labels)
columns_labels = ["pred_" + str(l) for l in labels]
index_labels = ["true_" + str(l) for l in labels]
cm = pd.DataFrame(cm,columns=columns_labels, index=index_labels)
print(cm.to_markdown())
ans_r = [c == a  for c,a in zip(y_pred,y_true)]
print(ans_r.count(True)/len(ans_r))
# %%
valid_dataset2 = [mn for mn in good_valid_dataset if mn.startswith("Marker")]
new_step2_dataset = step2_dataset[0] , valid_dataset2
# %%
model2 = make_model()
model2.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["binary_accuracy"])

tgen,vgen = make_generators(False,dataset_path,new_step2_dataset,batch_size,pick_func=pick)
tgen = from_generator(tgen)
vgen = from_generator(vgen)
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        min_lr=0.00001
                )
history2 = model2.fit(tgen,
        epochs=100, 
        batch_size=batch_size,
        validation_data= vgen,
        callbacks=[reduce_lr])
# %%
plot_history(history2.history,metrics=["binary_accuracy"])
plot_history(history2.history,metrics=["binary_accuracy"],is_loss=False)
# %%
labels = [1, 0]
_y_pred = model2.predict(vgen, verbose=1)
y_pred = [1 if p[0] > 0.5 else 0 for p in _y_pred]
y_true = []
for v in vgen:
    y_true +=list(v[1].numpy())
good_valid_dataset = valid_dataset[tp_flags]
cm = confusion_matrix(y_true, y_pred, labels=labels)
columns_labels = ["pred_" + str(l) for l in labels]
index_labels = ["true_" + str(l) for l in labels]
cm = pd.DataFrame(cm,columns=columns_labels, index=index_labels)
print(cm.to_markdown())
ans_r = [c == a  for c,a in zip(y_pred,y_true)]
print(ans_r.count(True)/len(ans_r))
# %%
