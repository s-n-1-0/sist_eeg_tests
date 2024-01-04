# %%
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from generator import RawGeneratorMaker,dataset_dir_path
from pickfunc import RawPickFuncMaker
import numpy as np
sample_size = 750
pfm = RawPickFuncMaker(sample_size)
maker = RawGeneratorMaker(f"{dataset_dir_path}/old_3pdataset.h5",valid_keys=[])
tgen,vgen = maker.make_generators(None,pick_func=pfm.make_random_pick_func())
# %%
csp_components = range(1,14)
scores = []

for n in csp_components:
    # CSPのインスタンス化
    csp = CSP(n_components=n)
    # LDA分類器のインスタンス化
    lda = LDA()
    # パイプラインの作成
    clf = Pipeline([('CSP', csp), ('LDA', lda)])

    # データをトレーニングセットとテストセットに分割
    x_train, y_train = tgen().__next__()
    x_train = x_train.transpose([0,2,1]).astype(np.float64)
    x_test,y_test = vgen().__next__()
    x_test = x_test.transpose([0,2,1]).astype(np.float64)
    x_train.shape,y_train.shape,x_test.shape,y_test.shape
    # パイプラインでトレーニング
    clf.fit(x_train, y_train)

    # テストデータで評価
    score = clf.score(x_test, y_test)
    scores.append(score)
scores