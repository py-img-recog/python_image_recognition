import copy

import torch
from torch import nn

class FNN(nn.Module):
    ''' Feed-forward neural networkモデル
    dim_input:         入力データの次元数
    dim_hidden:        隠れ層の次元数
    num_hidden_layers: 隠れ層の数
    num_classes:       識別対象の物体クラス数
    '''
    def __init__(self, dim_input, dim_hidden, num_hidden_layers, num_classes):
        # nn.Moduleクラスの初期化
        # nn.Moduleクラスを継承してモデルを実装する場合、必須
        super().__init__()

        # 隠れ層の数は1以上
        assert num_hidden_layers > 0

        # 隠れ層の生成
        self.layers = nn.ModuleList()
        self.layers.append(self._generate_hidden_layer(dim_input, dim_hidden))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(self._generate_hidden_layer(dim_hidden, dim_hidden))

        # 出力層の生成
        self.linear = nn.Linear(dim_hidden, num_classes)

    # 隠れ層生成関数
    def _generate_hidden_layer(self, dim_input, dim_output):
        # nn.Sequentialを使って複数処理を1つにまとめる
        layer = nn.Sequential(
            nn.Linear(dim_input, dim_output, bias=False),
            nn.BatchNorm1d(dim_output),
            nn.ReLU(inplace=True)
        )

        return layer

    # モデルパラメータのあるデバイスを返す関数
    def get_device(self):
        return self.linear.weight.device

    # モデルの複製関数
    def copy(self):
        return copy.deepcopy(self)

    # 入力に対する物体クラスの予測関数
    def forward(self, x, return_embed=False):
        h = x
        for layer in self.layers:
            h = layer(h)

        # 特徴量を返す
        if return_embed:
            return h

        y = self.linear(h)

        # ロジットを返す
        return y
