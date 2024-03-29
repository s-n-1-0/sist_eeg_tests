# %%
import tensorflow as tf
import keras.backend as K
from keras.utils import plot_model
from keras.metrics import Recall
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.layers import Dense,Activation,Dropout,Conv1D,MaxPooling1D,Flatten,BatchNormalization
import numpy as np
from generator import RawGeneratorMaker,dataset_dir_path
from pickfunc import RawPickFuncMaker
from summary import summary
from metrics import specificity
import pandas as pd
# %% 
offset = 500
sample_size = 750
csv_path = "./saves/mla_valid.csv"
valids = list(pd.read_csv(csv_path,header=None).to_numpy()[0,:])
sample_size = 750
pfm = RawPickFuncMaker(sample_size)
maker = RawGeneratorMaker(f"{dataset_dir_path}/merged_mla2.h5",valid_keys=[str(k) for k in valids])
save_path = f"./saves/3p/1dcnn_raw_{len(pfm.ch_list)}_{maker.split_mode}"#f"./saves/3p/dec2"
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
model.add(Dense(128,activation="sigmoid"))#note : not relu 何故か学習が進まない
model.add(Dropout(0.4))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["binary_accuracy",Recall(),specificity])
output_shapes=([None,sample_size,len(pfm.ch_list)], [None])

tgen,vgen = maker.make_generators(batch_size,pick_func=pfm.make_random_pick_func())
def from_generator(gen):
    return tf.data.Dataset.from_generator(gen,output_types=(np.float32,np.float32), output_shapes=output_shapes)
tgen = from_generator(tgen)
vgen = from_generator(vgen)
model.build(output_shapes[0])
model.summary()
plot_model(model,to_file="C://d/1dcnn.png",show_shapes=True)
# %%
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=4,
                        min_lr=0.00001
                )
checkpoint = ModelCheckpoint(
                    filepath=save_path+"/model-{epoch:02d}.h5",
                    monitor='val_loss',
                    save_best_only=True,
                )
history = model.fit(tgen,
        epochs=50, 
        batch_size=batch_size,
        validation_data= vgen,
        callbacks=[reduce_lr,checkpoint])
# %%
summary(model,history,vgen,save_path)
# %%
