# %%
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense,Activation,Dropout,Conv1D,MaxPooling1D,Flatten,BatchNormalization
import numpy as np
from generator import RawGeneratorMaker
from utils.history import save_history,plot_history
from sklearn.metrics import confusion_matrix
import pandas as pd

root_path = "D:/Dataset"
# %% 
offset = 0
back = 500
ch = 3
batch_size = 32
# %%
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
model.build(output_shapes[0])
model.summary()
# %%
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=4,
                        min_lr=0.00001
                )
history = model.fit(tgen,
        epochs=50, 
        batch_size=batch_size,
        validation_data= vgen,
        callbacks=[reduce_lr])
# %%
plot_history(history.history,metrics=["binary_accuracy"])
plot_history(history.history,metrics=["binary_accuracy"],is_loss=False)
save_history(".",history.history)

# %%
model.save("./saves/3p/model.h5",save_format="h5")
hist_df = pd.DataFrame(history.history)
hist_df.to_csv('./saves/3p/history.csv')
# %% predict
labels = [1, 0]
_y_pred = model.predict(vgen, verbose=1)
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