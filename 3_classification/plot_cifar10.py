import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torchvision

if __name__ == '__main__':
    # データセットの用意
    dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=True)

    # t-SNEのためにデータを整形
    x = []
    y = []
    num_samples = 500
    for i in range(num_samples):
        img, label = dataset[i]

        # 画像を平坦化 (32 x 32 x 3 -> 3027に変換)
        img_flatten = np.asarray(img).flatten()
        x.append(img_flatten)
        y.append(label)

    # 全データをNumPy配列に統合
    x = np.stack(x)
    y = np.array(y)

    # t-SNEを適用
    t_sne = TSNE(n_components=2, random_state=0)
    x_reduced = t_sne.fit_transform(x)

    # 各ラベルの色とマーカーを設定
    cmap = plt.get_cmap("tab10")
    markers = ['4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D']

    # データをプロット
    plt.figure(figsize=(20, 15))
    for i, cls in enumerate(dataset.classes):
        plt.scatter(x_reduced[y == i, 0], x_reduced[y == i, 1],
                    c=[cmap(i / len(dataset.classes))], marker=markers[i], s=100, alpha=0.8, label=cls)
    plt.axis('off')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=24)
    plt.show()
