import random

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
