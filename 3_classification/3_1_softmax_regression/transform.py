import numpy as np

import torchvision

# 画像整形用関数
def transform(img, channel_mean=None, channel_std=None):
    # PIL to numpy array, PyTorchでの処理用に単精度少数を使用
    img = np.asarray(img, dtype='float32')

    # 画像を平坦化
    x = img.flatten()

    # 入力の各次元を正規化
    if channel_mean is not None and channel_std is not None:
        x = (x - channel_mean) / channel_std

    return x

# ラベル整形用関数
def target_transform(label, num_classes=10):
    # 数字 -> One-hotに変換
    y = np.identity(num_classes)[label]

    return y

if __name__ == '__main__':
    # 学習、評価セットの用意
    dataset = torchvision.datasets.CIFAR10(root='data', download=True,
                                           transform=transform, target_transform=target_transform)

    # データのサンプルと表示
    img, label = dataset[0]
    print(f'Image: {img}')
    print(f'Label: {label}')
