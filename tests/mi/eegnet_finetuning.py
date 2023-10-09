# %%
import keras
import tensorflow as tf
from keras.metrics import Recall
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from metrics import specificity
from generator import RawGeneratorMaker,dataset_dir_path
from pickfunc import RawPickFuncMaker
import numpy as np
from summary import summary
import pandas as pd
model_path = "C:/_models/dec1/model-85.h5"
sample_size = 750
batch_size = 32
# %%
csv_path = "./saves/mla_valid.csv"
valids = list(pd.read_csv(csv_path,header=None).to_numpy()[0,:])
model = keras.models.load_model(model_path,custom_objects={"specificity":specificity})
pfm = RawPickFuncMaker(sample_size)
maker = RawGeneratorMaker(f"{dataset_dir_path}/merged_mla2.h5",valid_keys=[str(k) for k in valids])
save_path = f"./saves/3p/eegnet_raw_{len(pfm.ch_list)}_{maker.split_mode}_ft"
output_shapes=([None,len(pfm.ch_list),sample_size,1], [None])
model.summary()
# %%
tgen,vgen = maker.make_2d_generators(batch_size,pick_func=pfm.make_random_pick_func())
def from_generator(gen):
    return tf.data.Dataset.from_generator(gen,output_types=(np.float32,np.float32), output_shapes=output_shapes)
tgen = from_generator(tgen)
vgen = from_generator(vgen)
model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.0001), #0.000001
            metrics=["binary_accuracy",Recall(),specificity])

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
        epochs=2000, 
        batch_size=batch_size,
        validation_data= vgen,
        callbacks=[reduce_lr,checkpoint])
# %%
summary(model,history,vgen,save_path)
# %%
