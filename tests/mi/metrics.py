import keras.backend as K
#
# NOTE: 特異度関数
#

def specificity(y_true, y_pred):
    """
    特異度
    """
    # y_true: 正解ラベル
    # y_pred: 予測ラベル（確率ではなくクラスの予測値）
    # 予測ラベルをクラスに変換
    y_pred = K.round(y_pred)
    # Confusion matrixの計算
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    # 特異度の計算
    specificity = true_negatives / (true_negatives + false_positives + K.epsilon())

    return specificity