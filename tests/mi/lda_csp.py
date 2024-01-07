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
def build_csp_lda(n_components:int):
    csp = CSP(n_components=n_components)
    lda = LDA()
    # パイプラインの作成
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    return clf


# %%
csp_components = range(1,14)
scores = [[] for _ in range(13)]
for _ in range(100):
    for n in csp_components:
        clf = build_csp_lda(n)
        x_train, y_train = tgen().__next__()
        x_train = x_train.transpose([0,2,1]).astype(np.float64)
        x_test,y_test = vgen().__next__()
        x_test = x_test.transpose([0,2,1]).astype(np.float64)
        x_train.shape,y_train.shape,x_test.shape,y_test.shape
        # パイプラインでトレーニング
        clf.fit(x_train, y_train)

        # テストデータで評価
        score = clf.score(x_test, y_test)
        scores[n-1].append(score)
scores
# %%
save_path = "./saves/csp_lda_score.npy"
np.save(save_path,np.array(scores))

# %%
scores2 = np.load(save_path)
list(np.mean(scores2,axis=1)),list(np.std(scores2,axis=1))

# %% n=2の時のモデルを作成して保存
clf = build_csp_lda(n)
x_train, y_train = tgen().__next__()
x_train = x_train.transpose([0,2,1]).astype(np.float64)
x_test,y_test = vgen().__next__()
x_test = x_test.transpose([0,2,1]).astype(np.float64)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
# パイプラインでトレーニング
clf.fit(x_train, y_train)

# %% clfを保存
from joblib import dump
dump(clf, 'cspldf.joblib')

# %%
clf.predict(x_train)