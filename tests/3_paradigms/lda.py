
# %%
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from generator import dataset_dir_path,PsdGeneratorMaker
from pickfunc import PsdPickFuncMaker

# %% 関数化
pfm = PsdPickFuncMaker()

maker = PsdGeneratorMaker(dataset_dir_path+"/3pdataset.h5")
tgen,vgen = maker.make_generators(32,pfm.make_pick_func())

def merge_gen(gen):
    xd = np.zeros((0,50,len(pfm.ch_list)))
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
lda = LinearDiscriminantAnalysis() # SWLDA(shrinkage=0.5)
lda.fit(x_train, y_train)

#%%
# テストデータの予測
predictions = lda.predict(x_valid)

print(predictions)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# 混同行列の作成
cm = confusion_matrix(y_valid, predictions)
print(cm)
accuracy = accuracy_score(y_valid, predictions)
print("Accuracy:", accuracy)
# クラス1の予測精度の計算と出力
class1_predictions = predictions[y_valid == 0]
class1_accuracy = accuracy_score(y_valid[y_valid == 0], class1_predictions)
print("Class 1 Accuracy:", class1_accuracy)

# クラス2の予測精度の計算と出力
class2_predictions = predictions[y_valid == 1]
class2_accuracy = accuracy_score(y_valid[y_valid == 1], class2_predictions)
print("Class 2 Accuracy:", class2_accuracy)


# %%
coef = lda.coef_
intercept = lda.intercept_
np.save(f"./saves/3p/lda_psd_{len(pfm.ch_list)}_A/coef",coef)
np.save(f"./saves/3p/lda_psd_{len(pfm.ch_list)}_A/intercept",intercept)

# %%
