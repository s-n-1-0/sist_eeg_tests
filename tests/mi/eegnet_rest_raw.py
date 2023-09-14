#
# NOTE: まだRESTデータと結合してないけどHandデータ単体で学習はできる
#
# %%
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras import layers,Model,regularizers
from keras.constraints import max_norm
import numpy as np
from generator import RawGeneratorMaker,dataset_dir_path
from pickfunc import MultiRawPickFuncMaker
from summary import summary
# %% 
hand_sample_size = 750
rest_sample_size = 45
pfm = MultiRawPickFuncMaker(hand_sample_size=hand_sample_size,
                            rest_path=dataset_dir_path+"/rest.h5")
maker = RawGeneratorMaker(f"{dataset_dir_path}/merged.h5")
batch_size = 32
fs = 500

f1 = 24 #デフォ8
D = 2
kern_length = fs//2
dropout_rate = 0.5
norm_rate = 0.25
# %%
f2 = int(f1*D)
ch = len(pfm.ch_list)
rest_shape = (len(pfm.ch_list),rest_sample_size,1)
hand_shape = (len(pfm.ch_list),hand_sample_size,1)
def eegnet_build_branch(input_shape:tuple,name:str):
    branch_input = layers.Input(shape=input_shape,name=name)  # restデータの時間長とチャンネル数を指定
    branch = layers.Conv2D(f1, (1, kern_length), padding='same',
                        use_bias=False, kernel_regularizer=regularizers.l2(l=0.01))(branch_input)
    branch = layers.BatchNormalization()(branch)
    branch = layers.DepthwiseConv2D((ch, 1), use_bias=False,
                                depth_multiplier=D,
                                depthwise_constraint=max_norm(1.), kernel_regularizer=regularizers.l2(l=0.01))(branch)
    branch = layers.BatchNormalization()(branch)
    branch = layers.Activation('elu')(branch)
    branch = layers.AveragePooling2D((1, 4))(branch)
    branch = layers.Dropout(dropout_rate)(branch)

    branch = layers.SeparableConv2D(f2, (1, kern_length//4),
                                use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l=0.01))(branch)
    branch = layers.BatchNormalization()(branch)
    branch = layers.Activation('elu')(branch)
    branch = layers.AveragePooling2D((1, 8))(branch)
    branch = layers.Dropout(dropout_rate)(branch)
    branch = layers.Flatten()(branch)
    return branch_input,branch
def dnn_build_brunch(input_shape:tuple,name:str):
    branch_input = layers.Input(shape=input_shape,name=name)
    branch = layers.Flatten()(branch_input)
    return branch_input, layers.Dense(585, activation='sigmoid')(branch)
# ブランチの結合
hand_input,hand_branch = eegnet_build_branch(hand_shape,"input_1")
rest_input,rest_branch = dnn_build_brunch(rest_shape,"input_2")
combined = hand_branch#layers.concatenate([hand_branch]) #rest_branch
#combined = layers.Dense(64, activation='sigmoid')(combined)
#combined = layers.Dense(32, activation='sigmoid')(combined)
# 出力層
output = layers.Dense(1, activation='sigmoid',kernel_constraint=max_norm(norm_rate), kernel_regularizer=regularizers.l2(l=0.01))(combined)
model = Model(inputs=[hand_input], outputs=output) #, rest_input
model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["binary_accuracy"])
output_shapes=(([None,len(pfm.ch_list),hand_sample_size,1],
                [None,len(pfm.ch_list),rest_sample_size,1]),
                [None])

tgen,vgen = maker.make_2d_generators(batch_size,pick_func=pfm.make_random_pick_func(2000))
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
model.build(output_shapes[0])
model.summary()
# %%
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=10,
                        min_lr=0.0000000001
                )
history = model.fit(tgen,
        epochs=150, 
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
