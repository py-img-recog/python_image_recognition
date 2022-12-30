import torch
from torch import nn
from torchvision import models


class CNNEncoder(nn.Module):
    '''
    Transformer captioningのエンコーダ
    dim_embedding: 埋め込み次元
    '''
    def __init__(self, dim_embedding: int):
        super().__init__()

        # ImageNetで事前学習された
        # ResNet152モデルをバックボーンネットワークとする
        resnet = models.resnet152(weights="IMAGENET1K_V2") 

        # 特徴抽出器として使うため全結合層を削除
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        # デコーダへの出力
        self.linear = nn.Linear(resnet.fc.in_features, dim_embedding)

    '''
    エンコーダの順伝播
    imgs: 入力画像, [バッチサイズ, チャネル数, 高さ, 幅]
    '''
    def forward(self, imgs: torch.Tensor):
        # 特徴抽出 -> [バッチサイズ, 2048]
        # 今回はバックボーンネットワークは学習させない
        with torch.no_grad():
            features = self.backbone(imgs)
            features = features.flatten(1)

        # 全結合
        features = self.linear(features)

        return features
