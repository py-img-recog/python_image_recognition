import random

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

# Data loader生成関数
def generate_subset_sampler(dataset, ratio, random_seed=0):
    size = int(len(dataset) * ratio)

    indices = list(range(len(dataset)))

    # 学習セットと検証セットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)

    # 学習セットを学習とValidationに分割するSamplerを用意
    indices1, indices2 = indices[:size], indices[size:]
    sampler1 = SubsetRandomSampler(indices1)
    sampler2 = SubsetRandomSampler(indices2)

    return sampler1, sampler2

# データセットのデータの各次元の平均と標準偏差を計算する関数
def get_dataset_statistics(dataset):
    data = []
    for i in range(len(dataset)):
        img_flat = dataset[i][0]
        data.append(img_flat)
    data = np.stack(data)

    channel_mean = np.mean(data, axis=0)
    channel_std = np.std(data, axis=0)

    return channel_mean, channel_std

# t-SNEのプロット関数
def plot_t_sne(data_loader, model, num_samples=500):
    model.eval()

    # t-SNEのためにデータを整形
    x = []
    y = []
    for imgs, labels in data_loader:
        # データをモデルと同じデバイスに
        imgs = imgs.to(model.get_device())

        # 特徴量の抽出
        embeddings = model(imgs, return_embed=True)

        # detach関数で逆伝搬のグラフから分離
        x.append(embeddings.detach().to('cpu'))
        y.append(labels.clone().detach())

    # 全データを第1軸で結合
    x = torch.concat(x)
    y = torch.concat(y)

    # NumPy配列に変換
    x = x.numpy()
    y = y.numpy()

    # 指定サンプル数だけ抽出
    x = x[:num_samples]
    y = y[:num_samples]

    # t-SNEを適用
    t_sne = TSNE(n_components=2, random_state=0)
    x_reduced = t_sne.fit_transform(x)

    # 各ラベルの色とマーカーを設定
    cmap = plt.get_cmap("tab10")
    markers = ['4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D']

    # データをプロット
    plt.figure(figsize=(20, 15))
    for i, cls in enumerate(data_loader.dataset.classes):
      plt.scatter(x_reduced[y == i, 0], x_reduced[y == i, 1],
                  c=[cmap(i / len(data_loader.dataset.classes))], marker=markers[i], s=100, alpha=0.8, label=cls)
    plt.axis('off')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=24)
    plt.show()

if __name__ == '__main__':
    # 学習、評価セットの用意
    train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=True)

    # 学習セットのうち、検証セットに使う割合
    val_ratio = .2

    # Subset samplerの生成
    val_sampler, train_sampler = generate_subset_sampler(train_dataset, val_ratio)

    print(f'Train dataset size: {len(train_sampler)}')
    print(f'Validation dataset size: {len(val_sampler)}')
    print(f'Test dataset size: {len(test_dataset)}')

    # DataLoaderを生成
    train_loader = DataLoader(train_dataset, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, sampler=val_sampler)
    test_loader = DataLoader(test_dataset)
