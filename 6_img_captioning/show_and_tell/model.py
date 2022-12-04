import torch
from torch import nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    ''' Show and tellのエンコーダ
    embedding_dim: 埋め込み次元
    '''
    def __init__(self, embedding_dim: int):
        super(EncoderCNN, self).__init__()

        # IMAGENET1K_V2で事前学習された
        # ResNet152モデルをバックボーンとする
        resnet = models.resnet152(weights="IMAGENET1K_V2") 
        
        # 特徴抽出器として使うため全結合層を削除
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # デコーダへの出力
        self.linear = nn.Linear(resnet.fc.in_features, embedding_dim)

    '''
    エンコーダの順伝播
    images : 入力画像テンソル [バッチサイズ, チャネル数, 高さ, 幅]
    '''
    def forward(self, images: torch.Tensor):
        # 特徴抽出 -> [バッチサイズ, 2048, 1, 1]
        features = self.resnet(images) 

        # 形状変換 -> [バッチサイズ, 2048]
        features = features.reshape(features.size(0), -1)

        # 全結合
        features = self.linear(features)

        return features

class DecoderRNN(nn.Module):
    ''' Show and tellのデコーダ
    embedding_dim: 埋め込み次元（単語埋め込み次元）
    hidden_dim:    隠れ層次元
    vocab_size:    辞書サイズ
    num_layers:    レイヤ数
    max_seg_len:   最大系列帳
    dropout:       ドロップアウト確率
    '''
    def __init__(self, embedding_dim: int, hidden_dim: int, 
                 vocab_size: int, num_layers: int, 
                 max_seg_len: int = 20, dropout: int = 0.1):
        super(DecoderRNN, self).__init__()

        # 単語埋め込み
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                            num_layers, batch_first=True)

        # 単語IDへの線形変換
        self.linear = nn.Linear(hidden_dim, vocab_size)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

        # ソフトマックスによる確率分布予測
        self.softmax = nn.Softmax(dim=1)

    ''' デコーダの順伝播
    features: エンコーダ出力特徴 [バッチサイズ, 埋め込み次元]
    captions: 画像キャプション [バッチサイズ, 系列長]
    lengths:  系列長のリスト
    '''
    def forward(self, features: torch.Tensor, 
                captions: torch.Tensor, lengths: list):

        # 単語埋め込み -> [バッチサイズ, 系列長, 埋め込み次元]
        embeddings = self.embed(captions)

        # 画像埋め込みと単語埋め込みとを結合
        # features.unsqueeze(1) -> [バッチサイズ, 1, 埋め込み次元]
        # 結合後embeddings-> [バッチサイズ, 系列長+1, 埋め込み次元]
        embeddings = torch.cat((features.unsqueeze(1), 
                                embeddings), 1)
        
        # パディングされたTensorを可変長系列に戻してパック
        # packed.data() -> [実際の系列長, 埋め込み次元]
        packed = pack_padded_sequence(embeddings, 
                                      lengths, batch_first=True)

        # LSTM
        hiddens, cell = self.lstm(packed)

        # ドロップアウト
        output = self.dropout(hiddens[0])

        # 確率分布予測
        outputs = self.linear(output)

        return outputs

    ''' サンプリングによる説明文出力（ビームサーチ無し）
    features:   エンコーダ出力特徴 [バッチサイズ, 埋め込み次元]
    states:     LSTM隠れ状態
    max_length: キャプションの最大系列長
    '''
    def sample(self, features: torch.Tensor, 
               states=None, max_length=30):

        inputs = features.unsqueeze(1)
        word_idx_list = []

        # 最大系列長まで再帰的に単語をサンプリング予測
        for step_t in range(max_length):
            # LSTM隠れ状態を更新
            hiddens, states = self.lstm(inputs, states)

            # 単語予測
            outputs = self.linear(hiddens.squeeze(1))
            outputs = self.softmax(outputs)
            prob, predicted = outputs.max(1)
            word_idx_list.append(predicted[0].item())
            
            # t+1の入力を作成
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)  

        return word_idx_list