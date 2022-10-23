import argparse
from tqdm import tqdm
from collections import deque
import numpy as np

from torch.utils.data import DataLoader
import torchvision

import util
import transform
from model import SoftmaxRegression
import evaluator

# 引数のパース関数
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='検証に使う学習セット内のデータの割合')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='学習エポック数')
    parser.add_argument('--lrs', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4],
                        help='検証する学習率')
    parser.add_argument('--moving_avg', type=int, default=20,
                        help='移動平均で計算する損失と正確度の値の数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='ミニバッチサイズ')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='データローダーに使うCPUプロセスの数')

    return parser.parse_args()

def main(args):
    # 入力データ正規化のために学習セットのデータの各次元の平均と標準偏差を計算
    dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform.transform)
    channel_mean, channel_std = util.get_dataset_statistics(dataset)

    # 学習、評価セットの用意
    train_dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True,
        transform=lambda x: transform.transform(x, channel_mean, channel_std),
        target_transform=transform.target_transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True,
        transform=lambda x: transform.transform(x, channel_mean, channel_std),
        target_transform=transform.target_transform)

    # Subset samplerの生成
    val_sampler, train_sampler = util.generate_subset_sampler(train_dataset, args.val_ratio)

    print(f'Train dataset size: {len(train_sampler)}')
    print(f'Validation dataset size: {len(val_sampler)}')
    print(f'Test dataset size: {len(test_dataset)}')

    # DataLoaderを生成
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # 検証セットの結果による最良モデルの保存用変数
    val_loss_best = float('inf')
    model_best = None
    for lr in args.lrs:
        print(f'Learning rate: {lr}')

        # Softmax regressionモデルの生成
        model = SoftmaxRegression(32 * 32 * 3, len(train_dataset.classes))

        for epoch in range(args.num_epochs):
            with tqdm(train_loader) as pbar:
                pbar.set_description(f'[Epoch {epoch + 1}]')

                # Moving average計算用
                losses = deque()
                accs = deque()
                for x, y in pbar:
                    # サンプルしたデータはPyTorchのTensorのためNumPyデータに戻す
                    x = x.numpy()
                    y = y.numpy()

                    y_pred = model.predict(x)

                    # 学習データに対する損失と正確度を計算
                    loss = np.mean(np.sum(-y * np.log(y_pred), axis=1))
                    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

                    # 移動平均を計算して出力
                    losses.append(loss)
                    accs.append(accuracy)
                    if len(losses) > args.moving_avg:
                        losses.popleft()
                        accs.popleft()
                    pbar.set_postfix({'loss': np.mean(losses), 'accuracy': np.mean(accs)})

                    # パラメータを更新
                    model.update_parameters(x, y, y_pred, lr=lr)

            # 検証
            val_loss, val_accuracy = evaluator.evaluate(val_loader, model)
            print(f'Validation: loss = {val_loss:.3f}, accuracy = {val_accuracy:.3f}')

            # より良い検証結果が得られた場合、モデルを記録
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                model_best = model.copy()

    # テスト
    test_loss, test_accuracy = evaluator.evaluate(test_loader, model_best)
    print(f'      Test: loss = {test_loss:.3f}, accuracy = {test_accuracy:.3f}')

if __name__ == '__main__':
    args = get_args()
    main(args)
