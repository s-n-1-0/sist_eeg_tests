# %%
import keras
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from metrics import specificity
from generator import RawGeneratorMaker,dataset_dir_path
from pickfunc import RawPickFuncMaker
import numpy as np
from summary import summary
model_path = "C:/_models/dec2/model.h5"
sample_size = 750
batch_size = 32
# %%

model = keras.models.load_model(model_path,custom_objects={"specificity":specificity})
pfm = RawPickFuncMaker(sample_size)
maker = RawGeneratorMaker(f"{dataset_dir_path}/old_3pdataset.h5",valid_keys=[])
save_path = f"./saves/test/3p/1dcnn_raw_{len(pfm.ch_list)}_{maker.split_mode}_al"
output_shapes=([None,sample_size,len(pfm.ch_list)], [None])
model.summary()
# %%
tgen,vgen = maker.make_generators(batch_size,pick_func=pfm.make_random_pick_func())
def from_generator(gen):
    return tf.data.Dataset.from_generator(gen,output_types=(np.float32,np.float32), output_shapes=output_shapes)
tgen = from_generator(tgen)
vgen = from_generator(vgen)

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
        initial_epoch=50,
        epochs=100, 
        batch_size=batch_size,
        validation_data= vgen,
        callbacks=[reduce_lr,checkpoint])
# %%
summary(model,history,vgen,save_path)
# %%
