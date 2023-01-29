# %%
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense,Activation,Dropout,Conv1D,MaxPooling1D,Flatten,BatchNormalization
import numpy as np
from test2_generator import make_generators,make_test_generator
from utils.history import save_history,plot_history
from sklearn.metrics import confusion_matrix
import pandas as pd
# %% 
offset = 250
back = 750
ch = 10
batch_size = 32
# %%
model = Sequential()
model.add(Conv1D(
            filters=128,
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
model.add(Dense(3,activation="softmax"))
model.compile(loss='categorical_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["acc"])
output_shapes=([None,back,ch], [None,3])

def take6_pick(signal:np.ndarray,mode:bool):
    return signal[:,offset:back+offset]
    #return signal[:,:back]
tgen,vgen = make_generators("./dataset/lord2/train/ex.h5",batch_size,595,-216,pick_func=take6_pick)
def from_generator(gen):
    return tf.data.Dataset.from_generator(gen,output_types=(np.float32,np.float32), output_shapes=output_shapes)
tgen = from_generator(tgen)
vgen = from_generator(vgen)
model.build(output_shapes[0])
model.summary()
# %%
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=4,
                        min_lr=0.000001
                )
history = model.fit(tgen,
        epochs=100, 
        batch_size=batch_size,
        validation_data= vgen,
        callbacks=[reduce_lr])
# %%
plot_history(history.history,metrics=["acc"])
plot_history(history.history,metrics=["acc"],is_loss=False)
save_history(".",history.history)

# %%
model.save(".\model_e500.h5",save_format="h5")
# %% predict
def vec2label(x:list):
    if x[0] == 1:
        return "wait"
    elif x[1] == 1:
        return "light"
    elif x[2] == 1:
        return
labels = [0,1,2]
_y_pred = model.predict(vgen, verbose=1)
y_pred = [np.argmax(p)for p in _y_pred ]
x_valid = []
y_true = []
for v in vgen:
    x_valid += list(v[0].numpy())
    y_true += [np.argmax(y) for y in list(v[1].numpy())]
cm = confusion_matrix(y_true, y_pred, labels=labels)
columns_labels = ["pred_" + str(l) for l in labels]
index_labels = ["true_" + str(l) for l in labels]
cmdf = pd.DataFrame(cm,columns=columns_labels, index=index_labels)
print(cmdf.to_markdown())
print("")
cmrate = [cm[label,:] / y_true.count(label) for label in labels]
columns_labels = [f"pred_{str(l)}({str(y_true.count(l))})" for l in labels]
index_labels = [f"true_{str(l)}({str(y_true.count(l))})" for l in labels]
cmdf = pd.DataFrame(cmrate,columns=columns_labels, index=index_labels)
print(cmdf.to_markdown())
ans_r = [c == a  for c,a in zip(y_pred,y_true)]
print(ans_r.count(True)/len(ans_r))
# %%
