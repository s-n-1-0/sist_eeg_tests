"""
多ラベル用評価関数
コード引用 : https://qiita.com/persimmon-persimmon/items/b5ba97c92b54e2f6f469
"""
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import clip_ops,math_ops
from keras import backend as K

# 評価関数定義
# total_acc,binary_accは参考の「1つの画像が複数のクラスに属する場合（Multi-label）の画像分類」を参照
def total_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.prod(flag, axis=-1)

def binary_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.mean(flag, axis=-1)

# モデルがposiと予測して真の値がposiの割合
def precision(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.5), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.5), 'float32'))
    precision = true_positives / (pred_positives + K.epsilon())
    return precision

# 真の値がposiのもののうち、モデルがposiと予測した割合
def recall(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.5), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.5), 'float32'))
    recall = true_positives / (poss_positives + K.epsilon())
    return recall