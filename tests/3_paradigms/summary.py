
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils.history import plot_history, save_history

def summary(model,history,vgen,save_path:str):
    plot_history(history.history,metrics=["binary_accuracy"])
    plot_history(history.history,metrics=["binary_accuracy"],is_loss=False)
    
    save_history(save_path,history.history)
    model.save(f"{save_path}/model.h5",save_format="h5")

    labels = [1, 0]
    _y_pred = model.predict(vgen, verbose=1)
    y_pred = [1.0 if p[0] > 0.5 else 0 for p in _y_pred]
    x_valid = []
    y_true = []
    for v in vgen:
        x_valid += list(v[0].numpy())
        y_true +=list(v[1].numpy())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    columns_labels = ["pred_" + str(l) for l in labels]
    index_labels = ["true_" + str(l) for l in labels]
    cm = pd.DataFrame(cm,columns=columns_labels, index=index_labels)
    print(cm.to_markdown())
    ans_r = [c == a  for c,a in zip(y_pred,y_true)]
    print(ans_r.count(True)/len(ans_r))
