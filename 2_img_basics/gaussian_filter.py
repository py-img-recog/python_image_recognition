# 画像処理モジュールのインポート
from PIL import Image

# 数値計算モジュールのインポート
import numpy as np

# PILで読み込んだ画像をノートブックに表示するための関数をインポート
from IPython.display import display

if __name__ == '__main__':
    # グレースケール画像の読み込み
    img_gray = Image.open('drive/MyDrive/data/coffee.jpg')

    # グレースケール画像を表示
    display(img_gray)

    # グレースケール画像の配列サイズを表示
    print('Array size of the grayscale image: {}'.
            format(np.array(img_gray).shape))

    # グレースケール画像の原点の画素値を表示
    print('Pixel value of the grayscale image at (0, 0): {}'.
            format(img_gray.getpixel((0, 0))))

    # カラー画像の読み込み
    img_color = Image.open('drive/MyDrive/data/apple.jpg')

    # カラー画像を表示
    display(img_color)

    # カラー画像の配列サイズを表示
    print('Array size of the color image: {}'.
            format(np.array(img_color).shape))

    # カラー画像の原点の画素値を表示
    print('Pixel value of the color image at (0, 0): {}'.
            format(img_color.getpixel((0, 0))))
