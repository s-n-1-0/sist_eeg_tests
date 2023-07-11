
# %%
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from generator import dataset_dir_path,DwtGeneratorMaker
from pickfunc import DwtPickFuncMaker
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# %%
pfm = DwtPickFuncMaker()
maker = DwtGeneratorMaker(dataset_dir_path+"/3pdataset.h5")
tgen,vgen = maker.make_generators(32,pfm.make_pick_func())

def merge_gen(gen):
    xd = np.zeros((0,195,len(pfm.ch_list)))
    yd = np.zeros((0))
    for x,y in gen():
        xd = np.concatenate([xd,x],axis=0)
        yd = np.concatenate([yd,y],axis=0)

    #チャンネル結合
    xd = xd.reshape(xd.shape[0],-1)
    return xd,yd
x_train,y_train = merge_gen(tgen)
x_valid,y_valid = merge_gen(vgen)
x_train.shape,y_train.shape
# %%

#random.seed(41)
#X, y = random.shuffle(X, y)#, 
# 線形判別分析のモデルの初期化と学習
# SVMモデルのインスタンス化と訓練データへの適合
svm = SVC()
svm.fit(x_train, y_train)

# テストデータの予測
predictions = svm.predict(x_valid)
print(predictions)

# 混同行列の作成
cm = confusion_matrix(y_valid, predictions)
print(cm)

# 正確性の計算と表示
accuracy = accuracy_score(y_valid, predictions)
print("Accuracy:", accuracy)

# クラス1の予測精度の計算と表示
class1_predictions = predictions[y_valid == 0]
class1_accuracy = accuracy_score(y_valid[y_valid == 0], class1_predictions)
print("Class 1 Accuracy:", class1_accuracy)

# クラス2の予測精度の計算と表示
class2_predictions = predictions[y_valid == 1]
class2_accuracy = accuracy_score(y_valid[y_valid == 1], class2_predictions)
print("Class 2 Accuracy:", class2_accuracy)

# %%
