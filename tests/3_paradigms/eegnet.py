# %%
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras import layers
from keras.constraints import max_norm
import numpy as np
from generator import RawGeneratorMaker,dataset_dir_path
from pickfunc import RawPickFuncMaker
from summary import summary
# %% 
offset = 0
sample_size = 500
pfm = RawPickFuncMaker(sample_size)
batch_size = 32
fs = 500

f1 = 16 #デフォ8
D = 2
kern_length = fs//2
dropout_rate = 0.5
norm_rate = 0.25
# %%
f2 = int(f1*D)
ch = len(pfm.ch_list)
model = Sequential()
model.add(layers.Conv2D(f1, (1, kern_length), padding='same',
                        use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.DepthwiseConv2D((ch, 1), use_bias=False,
                                depth_multiplier=D,
                                depthwise_constraint=max_norm(1.)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('elu'))
model.add(layers.AveragePooling2D((1, 4)))
model.add(layers.Dropout(dropout_rate))

model.add(layers.SeparableConv2D(f2, (1, kern_length//4),
                                use_bias=False, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('elu'))
model.add(layers.AveragePooling2D((1, 8)))
model.add(layers.Dropout(dropout_rate))
model.add(layers. Flatten(name='flatten'))
model.add(layers.Dense(1, name='dense',
                        kernel_constraint=max_norm(norm_rate)))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["binary_accuracy"])
output_shapes=([None,len(pfm.ch_list),sample_size,1], [None])

maker = RawGeneratorMaker(f"{dataset_dir_path}/3pdataset.h5")
tgen,vgen = maker.make_generators(batch_size,pick_func=pfm.make_pick_func())
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
summary(model,history,vgen,f"./saves/3p/eegnet_raw_{len(pfm.ch_list)}_{maker.split_mode}")
# %%
tgen,vgen = maker.make_generators(batch_size,pick_func=pfm.make_pick_func())
tgen().__next__()
# %%
