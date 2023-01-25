# %%
import sys
import os
import pandas as pd

from common import *
from keras.models import load_model,Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from utils.history import plot_history
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from generator import make_generators,make_test_generator
def load_dataset_list(fn:str):
    with open(fn,"r") as f:
        lst = []
        for x in f:
            lst.append(x.rstrip("\n"))
    return lst

step1_model = load_model("./step1_model.h5")
step1_full_model = load_model("./step1_model2.h5")
step2_dataset = tuple(load_dataset_list(f"./step2_{i}.txt") for i in range(2))
# %% check model2
step1_valid = load_dataset_list("./step1_1.txt")
step1_full_model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["binary_accuracy"])
test_gen = make_test_generator(is_step1=True,
                                path=dataset_path,
                                dataset=(None,step1_valid),
                                batch_size=batch_size,
                                pick_func=pick)
test_gen = from_generator(test_gen)
labels = [1, 0]
_y_pred = step1_full_model.predict(test_gen, verbose=1)
y_pred = [1.0 if p[0] > 0.5 else 0 for p in _y_pred]
x_valid = []
y_true = []
for v in test_gen:
    x_valid += list(v[0].numpy())
    y_true +=list(v[1].numpy())
cm = confusion_matrix(y_true, y_pred, labels=labels)
columns_labels = ["pred_" + str(l) for l in labels]
index_labels = ["true_" + str(l) for l in labels]
cm = pd.DataFrame(cm,columns=columns_labels, index=index_labels)
print(cm.to_markdown())
ans_r = [c == a  for c,a in zip(y_pred,y_true)]
print(ans_r.count(True)/len(ans_r))

# %% make step2 model
step1_model.trainable = False
model = Sequential()
model.add(step1_model)
model.add(Flatten())
model.add(Dense(128,activation="sigmoid"))
model.add(Dropout(0.4))
model.add(Dense(128,activation="sigmoid"))
model.add(Dropout(0.4))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='binary_crossentropy', 
            optimizer=tf.optimizers.Adam(learning_rate=0.001), #0.000001
            metrics=["binary_accuracy"])

tgen,vgen = make_generators(False,dataset_path,step2_dataset,batch_size,pick_func=pick)
tgen = from_generator(tgen)
vgen = from_generator(vgen)
model.build(output_shapes[0])
model.summary()
# %%
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.25,
                        patience=10,
                        min_lr=0.00001
                )
history = model.fit(tgen,
        epochs=500, 
        batch_size=batch_size,
        validation_data= vgen,
        callbacks=[reduce_lr])
# %%
plot_history(history.history,metrics=["binary_accuracy"])
plot_history(history.history,metrics=["binary_accuracy"],is_loss=False)
# %%
