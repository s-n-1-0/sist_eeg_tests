###
### 学習に使うデータセットの量で精度がどれほど変わるか確認コード(mnist)
###
# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from utils.history import plot_history
# %%
mnist =  fetch_openml('mnist_784')
x, y = mnist['data'].values, mnist['target']
x = x.reshape(x.shape[0], 28, 28,1).astype('float32') / 255.0

# %%

size1 = 60000
size2 = 70000
x_train,x_valid,y_train,y_valid = x[:size1],x[size1:size2],y[:size1],y[size1:size2]

y_train = np_utils.to_categorical(y_train, 10)
y_valid = np_utils.to_categorical(y_valid, 10)
# %%
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10,validation_data=(x_valid,y_valid))
# %%
plot_history(history.history,metrics=["accuracy"],is_loss=False)
# %%
