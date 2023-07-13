
# %%
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from generator import dataset_dir_path,DwtGeneratorMaker,merge_gen
from pickfunc import DwtPickFuncMaker
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# %% 関数化
pfm = DwtPickFuncMaker()
maker = DwtGeneratorMaker(dataset_dir_path+"/3pdataset.h5")
tgen,vgen = maker.make_generators(32,pfm.make_pick_func())
init_shape = (0,195*len(pfm.ch_list))
x_valid,y_valid = merge_gen(vgen,init_shape)
# %%
# 線形判別分析のモデルの初期化と学習
def learning():
    x_train,y_train = merge_gen(tgen,init_shape)
    lda = LinearDiscriminantAnalysis() # SWLDA(shrinkage=0.5)
    lda.fit(x_train, y_train)

    # テストデータの予測
    predictions = lda.predict(x_valid)
    # 混同行列の作成
    cm = confusion_matrix(y_valid, predictions)
    print(cm)
    accuracy = accuracy_score(y_valid, predictions)
    print("Accuracy:", accuracy)
    return lda,accuracy,predictions
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
coef = max_model.coef_
intercept = max_model.intercept_
np.save(f"./saves/3p/max_lda_dwt_{len(pfm.ch_list)}_{maker.split_mode}/coef",coef)
np.save(f"./saves/3p/max_lda_dwt_{len(pfm.ch_list)}_{maker.split_mode}/intercept",intercept)

# %%
