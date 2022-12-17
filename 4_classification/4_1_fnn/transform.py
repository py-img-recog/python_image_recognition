from PIL import Image
import numpy as np


'''
img         : 整形対称の画像
channel_mean: 各次元のデータセット全体の平均, [入力次元]
channel_std : 各次元のデータセット全体の標準偏差, [入力次元]
'''
def transform(img: Image.Image, channel_mean: np.ndarray=None,
              channel_std: np.ndarray=None):
    # PIL to numpy array, PyTorchでの処理用に単精度少数を使用
    img = np.asarray(img, dtype='float32')

    # [32, 32, 3]の画像を3072次元のベクトルに平坦化
    x = img.flatten()

    # 各次元をデータセット全体の平均と標準偏差で正規化
    if channel_mean is not None and channel_std is not None:
        x = (x - channel_mean) / channel_std

    return x
