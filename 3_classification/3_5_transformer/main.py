import argparse
from tqdm import tqdm
from collections import deque

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import util
from model import VisionTransformer
import evaluator

# 引数のパース関数
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='検証に使う学習セット内のデータの割合')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='パッチサイズ')
    parser.add_argument('--dim_hidden', type=int, default=512,
                        help='隠れ層の次元数')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Attention機構のヘッド数')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                        help='Transformer層のFNNの隠れ層の特徴次元数')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Transformerエンコーダ層の数')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='学習エポック数')
    parser.add_argument('--lrs', type=float, nargs='+',
                        default=[1e-2, 1e-3, 1e-4],
                        help='検証する学習率')
    parser.add_argument('--moving_avg', type=int, default=20,
                        help='移動平均で計算する損失と正確度の値の数')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='t-SNEでプロットするサンプル数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='ミニバッチサイズ')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='データローダーに使うCPUプロセスの数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='学習を行うデバイス')

    return parser.parse_args()

def main(args):
    # 正規化のため学習セットのデータで各チャネルの平均と標準偏差を計算
    dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True, transform=T.ToTensor())
    channel_mean, channel_std = util.get_dataset_statistics(dataset)

    transforms = T.Compose((
        T.ToTensor(),
        T.Normalize(mean=channel_mean, std=channel_std),
    ))

    # 学習、評価セットの用意
    train_dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True, transform=transforms)
    test_dataset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transforms)

    # Subset samplerの生成
    val_sampler, train_sampler = util.generate_subset_sampler(
        train_dataset, args.val_ratio)

    print(f'Train dataset size: {len(train_sampler)}')
    print(f'Validation dataset size: {len(val_sampler)}')
    print(f'Test dataset size: {len(test_dataset)}')

    # DataLoaderを生成
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # 目的関数の生成
    loss_func = F.cross_entropy

    # 検証セットの結果による最良モデルの保存用変数
    val_loss_best = float('inf')
    model_best = None
    for lr in args.lrs:
        print(f'Learning rate: {lr}')

        # Vision Transformerモデルの生成
        model = VisionTransformer(
            len(train_dataset.classes), 32, args.patch_size,
            args.dim_hidden, args.num_heads, args.dim_feedforward,
            args.num_layers)

        # モデルを指定デバイスに転送
        model.to(args.device)

        # Optimizerの生成
        optimizer = optim.SGD(model.parameters(), lr=lr)

        for epoch in range(args.num_epochs):
            # モデルを学習モードに設定
            # (バッチ正規化など、学習と推論で異なる処理をする場合、学習用の処理を実行)
            model.train()

            with tqdm(train_loader) as pbar:
                pbar.set_description(f'[Epoch {epoch + 1}]')

                # Moving average計算用
                losses = deque()
                accs = deque()
                for x, y in pbar:
                    # データをモデルと同じデバイスに転送
                    x = x.to(model.get_device())
                    y = y.to(model.get_device())

                    # パラメータの勾配をリセット
                    optimizer.zero_grad()

                    # 順伝搬
                    y_pred = model(x)

                    # 学習データに対する損失と正確度を計算
                    loss = loss_func(y_pred, y)
                    accuracy = (y_pred.argmax(dim=1) == y).float().mean()

                    # 誤差逆伝搬
                    loss.backward()

                    # パラメータの更新
                    optimizer.step()

                    # 移動平均を計算して出力
                    losses.append(loss.item())
                    accs.append(accuracy.item())
                    if len(losses) > args.moving_avg:
                        losses.popleft()
                        accs.popleft()
                    pbar.set_postfix({'loss': torch.Tensor(losses).mean().item(),
                                      'accuracy': torch.Tensor(accs).mean().item()})

            # 検証
            val_loss, val_accuracy = evaluator.evaluate(val_loader, model, loss_func)
            print(f'Validation: loss = {val_loss:.3f}, accuracy = {val_accuracy:.3f}')

            # より良い検証結果が得られた場合、モデルを記録
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                model_best = model.copy()

    # テスト
    test_loss, test_accuracy = evaluator.evaluate(test_loader, model_best, loss_func)
    print(f'      Test: loss = {test_loss:.3f}, accuracy = {test_accuracy:.3f}')

    # t-SNEで特徴量をプロット
    util.plot_t_sne(test_loader, model_best, args.num_samples)

if __name__ == '__main__':
    args = get_args()
    main(args)
