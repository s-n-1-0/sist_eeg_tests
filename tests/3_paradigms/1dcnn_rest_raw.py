# %%
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras import layers,Model
import numpy as np
from generator import RawGeneratorMaker,dataset_dir_path
from pickfunc import MultiRawPickFuncMaker
from summary import summary
# %% 
offset = 500
hand_sample_size = 1000
rest_sample_size = 5000
pfm = MultiRawPickFuncMaker(hand_sample_size=hand_sample_size,
                            rest_sample_size=rest_sample_size)
batch_size = 32
# %%
rest_shape = (rest_sample_size,len(pfm.ch_list))
hand_shape = (hand_sample_size,len(pfm.ch_list))
def build_branch(input_shape:tuple,name:str):#(rest_time_length, rest_channels)
    branch_input = layers.Input(shape=input_shape,name=name)  # restデータの時間長とチャンネル数を指定
    branch = layers.Conv1D(
            filters=128,
            kernel_size= 50,
            strides=5
            )(branch_input)
    branch = layers.BatchNormalization()(branch)
    branch = layers.Activation("relu")(branch)
    branch = layers.Dropout(0.4)(branch)
    branch = layers.Conv1D(
            filters=64,
            kernel_size= 5,
            strides=2
            )(branch)
    branch = layers.BatchNormalization()(branch)
    branch = layers.Activation("relu")(branch)
    branch = layers.Dropout(0.4)(branch)
    branch = layers.Conv1D(
            filters=128,
            kernel_size= 3,
            padding='same'
            )(branch)
    branch = layers.BatchNormalization()(branch)
    branch = layers.Activation("relu")(branch)
    branch = layers.Dropout(0.4)(branch)
    branch = layers.MaxPooling1D(
            pool_size=3,
            padding="same"
            )(branch)
    branch = layers.Flatten()(branch)
    branch = layers.Dense(64, activation='relu')(branch)
    branch = layers.Dense(32, activation='relu')(branch)
    return branch_input,branch
# ブランチの結合
rest_input,rest_branch = build_branch(rest_shape,"input_1")
hand_input,hand_branch = build_branch(hand_shape,"input_2")
combined = layers.concatenate([rest_branch,hand_branch])
combined = layers.Dense(64, activation='relu')(combined)
combined = layers.Dense(32, activation='relu')(combined)
# 出力層
output = layers.Dense(1, activation='sigmoid')(combined)
model = Model(inputs=[rest_input, hand_input], outputs=output)
model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["binary_accuracy"])
output_shapes=(([None,rest_sample_size,len(pfm.ch_list)],
                [None,hand_sample_size,len(pfm.ch_list)]), [None])

maker = RawGeneratorMaker(f"{dataset_dir_path}/3pdataset.h5")
tgen,vgen = maker.make_generators(batch_size,pick_func=pfm.make_random_pick_func(500,2000,rest_offset=1000))
def from_generator(gen):
    return tf.data.Dataset.from_generator(gen,output_signature=(
        {
            "input_1":tf.TensorSpec(shape=output_shapes[0][0], dtype=tf.float32),
            "input_2":tf.TensorSpec(shape=output_shapes[0][1], dtype=tf.float32)
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32))
    )

tgen = from_generator(tgen)
vgen = from_generator(vgen)
#model.build(output_shapes[0])
#model.summary()
# %%
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=4,
                        min_lr=0.00001
                )
history = model.fit(tgen,
        epochs=300, 
        batch_size=batch_size,
        validation_data= vgen,
        callbacks=[reduce_lr])
# %%
summary(model,history,vgen,f"./saves/3p/1dcnn_rest_raw_{len(pfm.ch_list)}_{maker.split_mode}")
# %%
tgen,vgen = maker.make_generators(batch_size,pick_func=pfm.make_pick_func())
for t in tgen():
    print(len(t))
# %%
