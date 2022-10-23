import torchvision

if __name__ == '__main__':
    # データセットの用意
    dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True)

    # 表示済みの画像のラベルを保存する変数
    displayed_classes = set()
    i = 0
    # 全てのラベルの画像を1枚ずつ抽出するまでループ
    while i < len(dataset) and len(displayed_classes) < len(dataset.classes):
        # インデクシングによりデータのサンプルが可能
        img, label = dataset[i]
        if label not in displayed_classes:
            print(f'物体クラス: {dataset.classes[label]}')

            # 元画像が小さいので、リサイズして表示
            img = img.resize((256, 256))
            display(img)

            # 表示済みラベルの追加
            displayed_classes.add(label)

        i += 1

