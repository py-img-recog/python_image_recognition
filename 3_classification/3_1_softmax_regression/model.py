import numpy as np

class SoftmaxRegression:
    ''' Softmax regressionモデル
    input_dim:   入力データの次元数
    num_classes: 識別対象の物体クラス数
    '''

    def __init__(self, input_dim, num_classes):
        # パラメータの初期化
        self.weight = np.random.uniform(
            low=-1 / input_dim ** .5, high=1 / input_dim ** .5, size=(input_dim, num_classes))
        self.bias = np.zeros(num_classes)

    # 内部用Softmax関数
    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    # モデルの複製関数
    def copy(self):
        model_copy = self.__class__(*self.weight.shape)
        model_copy.weight = self.weight.copy()
        model_copy.bias = self.bias.copy()

        return model_copy

    # 入力に対する物体クラスの予測関数
    def predict(self, x):
        # 入力の次元: batch_size * input_dim
        # 重み weightの次元: input_dim * num_classes
        # バイアス biasの次元: num_classes
        y = np.matmul(x, self.weight) + self.bias
        y = self._softmax(y)

        return y

    # パラメータ更新用関数
    def update_parameters(self, x, y, y_pred, lr=0.001):
        # 出力と正解の誤差を計算
        diffs = y_pred - y

        # パラメータを更新
        self.bias -= lr * np.mean(diffs, axis=0)
        self.weight -= lr * np.mean(x[:, :, np.newaxis] * diffs[:, np.newaxis], axis=0)

if __name__ == '__main__':
    # モデルの生成、初期化
    # 入力次元 = 32 * 32 * 3 = 3072, 物体クラス数 = 10
    model = SoftmaxRegression(32 * 32 * 3, 10)

    # ミニバッチサイズ1で入力をランダムに生成
    x = np.random.uniform(0., 1., size=(1, 32 * 32 * 3))

    # 予測
    y = model.predict(x)

    print(f'Prediction result: {y[0]}')
