import torch

# 評価用関数
def evaluate(data_loader, model, loss_func):
    # モデルを評価モードに設定
    # (バッチ正規化など、学習と推論で異なる処理をする場合、推論用の処理を実行)
    model.eval()

    losses = []
    preds = []
    for x, y in data_loader:
        with torch.no_grad():
            # データをモデルと同じデバイスに転送
            x = x.to(model.get_device())
            y = y.to(model.get_device())

            y_pred = model(x)

            losses.append(loss_func(y_pred, y, reduction='none'))

            # Maxのインデックス＝クラスラベル
            preds.append(y_pred.argmax(dim=1) == y)

    losses = torch.cat(losses, dim=0)
    preds = torch.cat(preds, dim=0)

    return losses.mean(), preds.float().mean()
