from PIL import Image
import numpy as np
from IPython.display import display

# 数値計算モジュールのインポート
from scipy import signal

def generate_kernel():
    # カーネル用の変数を用意
    kernel_h = np.zeros((3, 3))
    kernel_v = np.zeros((3, 3))
    kernel_lap = np.zeros((3, 3))

    # カーネルの値を設定
    kernel_h[1, 1] = -1
    kernel_h[1, 2] = 1
    kernel_v[1, 1] = -1
    kernel_v[2, 1] = 1
    kernel_lap[0, 1] = 1
    kernel_lap[1, 0] = 1
    kernel_lap[1, 2] = 1
    kernel_lap[2, 1] = 1
    kernel_lap[1, 1] = -4

    return kernel_h, kernel_v, kernel_lap

if __name__ == '__main__':
    # 画像の読み込み
    img = Image.open('drive/MyDrive/data/coffee.jpg')

    # NumPyとSciPyを使うため画像をNumPy配列に変換
    img = np.asarray(img, dtype='int32')

    # 一次微分のカーネル
    kernel_h, kernel_v, kernel_lap = generate_kernel()

    # 畳み込み演算
    img_h_diff = signal.convolve2d(img, kernel_h, 
                                   boundary='symm', mode='same')
    img_v_diff = signal.convolve2d(img, kernel_v, 
                                   boundary='symm', mode='same')
    img_lap = signal.convolve2d(img, kernel_lap, 
                                boundary='symm', mode='same')

    # 微分値の絶対値を計算
    img_h_diff = np.absolute(img_h_diff)
    img_v_diff = np.absolute(img_v_diff)

    # 水平一次微分画像と垂直一次微分画像の合成
    img_diff = (img_h_diff ** 2 + img_v_diff ** 2) ** 0.5

    # 範囲を超えた画素値をクリップ
    img_h_diff = np.clip(img_h_diff, 0, 255).astype('uint8')
    img_v_diff = np.clip(img_v_diff, 0, 255).astype('uint8')
    img_diff = np.clip(img_diff, 0, 255).astype('uint8')
    img_lap = np.clip(img_lap, 0, 255).astype('uint8')

    # NumPy配列をPIL画像に変換
    img_h_diff = Image.fromarray(img_h_diff)
    img_v_diff = Image.fromarray(img_v_diff)
    img_diff = Image.fromarray(img_diff)
    img_lap = Image.fromarray(img_lap)

    display(img_h_diff)
    display(img_v_diff)
    display(img_diff)
    display(img_lap)
