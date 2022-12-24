import random
import numpy as np

from torch.utils.data import Dataset


'''
dataset    : 分割対象のデータセット
ratio      : 1つ目のセットに含めるデータ量の割合
random_seed: 分割結果を不変にするためのシード
'''
def generate_subset(dataset: Dataset, ratio: float,
                    random_seed: int=0):
    # サブセットの大きさを計算
    size = int(len(dataset) * ratio)

    indices = list(range(len(dataset)))

    # 二つのセットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)

    # セット1とセット2のサンプルのインデックスに分割
    indices1, indices2 = indices[:size], indices[size:]

    return indices1, indices2


'''
dataset: 平均と標準偏差を計算する対象のPyTorchのデータセット
'''
def get_dataset_statistics(dataset: Dataset):
    data = []
    for i in range(len(dataset)):
        # 3072次元のベクトルを取得
        img_flat = dataset[i][0]
        data.append(img_flat)
    # 第0軸を追加して第0軸でデータを連結
    data = np.stack(data)

    # データ全体の平均と標準偏差を計算
    channel_mean = np.mean(data, axis=0)
    channel_std = np.std(data, axis=0)

    return channel_mean, channel_std
