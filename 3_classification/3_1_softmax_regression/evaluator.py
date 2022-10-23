import numpy as np

# 評価用関数
def evaluate(data_loader, model):
    losses = []
    preds = []
    for x, y in data_loader:
        # サンプルしたデータはPyTorchのTensorのためNumPyデータに戻す
        x = x.numpy()
        y = y.numpy()

        y_pred = model.predict(x)
      
        losses.append(-y * np.log(y_pred))

        # Maxのインデックス＝クラスラベル
        preds.append(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

    losses = np.concatenate(losses)
    preds = np.concatenate(preds)

    return np.mean(losses), np.mean(preds)

