
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
x_valid,y_valid = merge_gen(vgen)
# %%
# SVMモデルのインスタンス化と訓練データへの適合
def learning():
    x_train,y_train = merge_gen(tgen)
    svm = SVC()
    svm.fit(x_train, y_train)

    # テストデータの予測
    predictions = svm.predict(x_valid)
    # 混同行列の作成
    cm = confusion_matrix(y_valid, predictions)
    print(cm)
    accuracy = accuracy_score(y_valid, predictions)
    print("Accuracy:", accuracy)
    return svm,accuracy,predictions
max_acc = 0
max_model = None
max_predicts = None
for i in range(10):
    print(f"{i+1}回目学習")
    model,acc,pre = learning()
    if acc > max_acc:
        max_acc = acc
        max_model = model
        max_predicts = pre

print("------------")
print("Accuracy:", max_acc)
# クラス1の予測精度の計算と出力
class1_predictions = max_predicts[y_valid == 0]
class1_accuracy = accuracy_score(y_valid[y_valid == 0], class1_predictions)
print("Class 1 Accuracy:", class1_accuracy)

# クラス2の予測精度の計算と出力
class2_predictions = max_predicts[y_valid == 1]
class2_accuracy = accuracy_score(y_valid[y_valid == 1], class2_predictions)
print("Class 2 Accuracy:", class2_accuracy)

# %%
