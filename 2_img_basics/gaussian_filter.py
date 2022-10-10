from PIL import Image
import numpy as np
from IPython.display import display

# ガウシアンカーネルの生成関数
def generate_gaussian_kernel(kernel_width, kernel_height, sigma):
    # カーネルの大きさを奇数に限定に限定
    assert kernel_width % 2 == 1 and kernel_height % 2 == 1

    # カーネル用の変数を用意
    kernel = np.empty((kernel_height, kernel_width))

    for y in range(-(kernel_height // 2), kernel_height // 2 + 1):
        for x in range(-(kernel_width // 2), kernel_width // 2 + 1):
            # ガウス分布から値を抽出しカーネルに代入
            h = np.exp(-(x ** 2 + y  ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
            kernel[y + kernel_height // 2, x + kernel_width // 2] = h

    # カーネルの和が1になるように正規化
    kernel /= np.sum(kernel)

    return kernel

def apply_filter(img, kernel):
    # 画像サイズとカーネルサイズの取得
    width, height = img.size

    # フィルタ適用後の画像を保持する変数を用意
    img_filtered = Image.new(mode='L', size=(width, height))

    # フィルタ適用後の各画素値の計算
    for y in range(height):
        for x in range(width):
            filtered_value = convolution(img, kernel, x, y)
            img_filtered.putpixel((x, y), int(filtered_value))

    return img_filtered

def convolution(img, kernel, x, y):
    # 画像サイズとカーネルサイズの取得
    width, height = img.size
    kernel_height, kernel_width = kernel.shape[:2]

    # 畳み込み演算
    value = 0
    for y_kernel in range(-(kernel_height // 2), kernel_height // 2 + 1):
        for x_kernel in range(-(kernel_width // 2), kernel_width // 2 + 1):
            x_img = max(min(x + x_kernel, width - 1), 0)
            y_img = max(min(y + y_kernel, height - 1), 0)
            h = kernel[y_kernel + kernel_height // 2, x_kernel + kernel_width // 2]
            value += h * img.getpixel((x_img, y_img))

    return value

if __name__ == '__main__':
    # 画像の読み込み
    img = Image.open('drive/MyDrive/data/coffee_noise.jpg')

    # ガウシアンカーネルの生成
    kernel = generate_gaussian_kernel(kernel_width=5, kernel_height=5, sigma=1.3)

    # カーネルの表示
    print('Generated kernel:')
    print(kernel)

    # ガウシアンフィルタの適用
    img_filtered = apply_filter(img, kernel)

    # 元画像とフィルタ適用後の画像の表示
    display(img)
    display(img_filtered)
    
