import torch
from torch import nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence

# GPUの設定
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

class EncoderCNN(nn.Module):
    ''' Show, attend and tellのエンコーダ
    encoded_image_size: 画像部分領域サイズ
    embedding_dim:      埋込みの次元
    '''
    def __init__(self, encoded_image_size: int, 
                 embedding_dim: int):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size

        # IMAGENET1K_V2で事前学習された
        # ResNet152モデルをバックボーンとする
        resnet = models.resnet152(weights="IMAGENET1K_V2") 
        
        # プーリング層と全結合層を削除
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # AdaptiveAvgPool2dで部分領域(14x14)を作成
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
                                (encoded_image_size, 
                                 encoded_image_size))

    ''' エンコーダの順伝播
    images : 入力画像テンソル [バッチサイズ, チャネル数, 高さ, 幅]
    '''
    def forward(self, images: torch.Tensor):
        # 特徴抽出
        features = self.resnet(images) 
        features = self.adaptive_pool(features)

        # 並び替え -> [バッチサイズ, 14, 14, 2048]
        features = features.permute(0, 2, 3, 1)

        return features

class Attention(nn.Module):
    ''' 注意機構 (Attention mechanism)
    encoder_dim: エンコーダ出力の特徴次元
    decoder_dim: デコーダ出力の次元
    attention_dim: 注意機構の次元
    '''
    def __init__(self, encoder_dim: int, 
                 decoder_dim: int, attention_dim: int):
        super(Attention, self).__init__()

        # z: エンコーダ出力を変換する線形層(Wz)
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)

        # h: デコーダ出力を変換する線形層(Wh)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)

        # f: zとhを足すための線形層
        self.full_att = nn.Linear(attention_dim, 1)

        # α: アテンション重みを計算するソフトマックス層
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    ''' Attentionの順伝播
    encoder_out: エンコーダ出力
    decoder_hidden: デコーダ隠れ状態の次元
    '''
    def forward(self, encoder_out: torch.Tensor, 
                decoder_hidden: torch.Tensor):

        # e: アライメントスコア
        att1 = self.encoder_att(encoder_out) # Wz * z
        att2 = self.decoder_att(decoder_hidden) # Wh * h_{t-1}
        att = self.full_att(
                self.relu(att1 + att2.unsqueeze(1))).squeeze(2) 

        # α: T個の部分領域ごとのアテンション重み
        alpha = self.softmax(att)

        # c: コンテキストベクトル
        context_vector = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return context_vector, alpha


class DecoderWithAttention(nn.Module):
    ''' 注意機構 (Attention mechanism)付きデコーダネットワーク
    attention_dim: 注意機構の次元
    embed_dim: 埋込み次元
    decoder_dim: デコーダの次元
    vocab_size: 辞書の次元
    encoder_dim: エンコーダ出力の特徴次元
    '''
    def __init__(self, attention_dim: int, embed_dim: int, 
                 decoder_dim: int, vocab_size: int, 
                 encoder_dim: int=2048, dropout: float=0.5):
        super(DecoderWithAttention, self).__init__()

        # パラメータ
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # 注意機構
        self.attention = Attention( encoder_dim, 
                                    decoder_dim, 
                                    attention_dim)

        # 単語の埋め込み
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)

        # LSTMセル
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, 
                                       decoder_dim, bias=True)

        # LSTM隠れ状態/メモリセルを初期化
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

         # シグモイド活性化前の線形層
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # 単語出力用の線形層
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # 埋め込み層、全結合層の重みを初期化
        self.init_weights()
        
    '''
    デコーダの重みパラメータを初期化
    '''
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    ''' 画像特徴の平均値で隠れ状態とメモリセルを初期化
    encoder_out: エンコーダ出力 [バッチサイズ, 14, 14, 2048]
    '''
    def init_hidden_state(self, encoder_out: torch.Tensor):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    ''' 注意機構 (Attention mechanism)付きデコーダのForward pass
    encoder_out: エンコーダ出力 [バッチサイズ, 14, 14, 2048]
    encoded_captions: キャプション [バッチサイズ, 最大系列長]
    caption_lengths: 系列長 [バッチサイズ, 1]
    '''
    def forward(self, encoder_out: torch.Tensor, 
                encoded_captions: torch.Tensor,
                caption_lengths: list):
        # パラメータ
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # 画像を平坦化
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) 
        num_pixels = encoder_out.size(1)

        # 単語埋込み
        embedded_captions = self.embedding(encoded_captions)

        # 隠れ状態ベクトル、メモリセルを初期化
        h, c = self.init_hidden_state(encoder_out)

        # 最大系列長
        # <end>を除くため(caption_lengths - 1)とする
        caption_lengths = torch.tensor(caption_lengths)
        dec_lengths = (caption_lengths - 1).tolist()

        # キャプショニング結果を保持するためのテンソル
        predictions = torch.zeros(batch_size, 
                                    max(dec_lengths), 
                                    vocab_size).to(device)

        # アテンション重みを保持するためのテンソル
        alphas = torch.zeros(batch_size, 
                                max(dec_lengths), 
                                num_pixels).to(device)

        # t-1のデコーダ隠れ状態出力をアテンション重み付けして復号化
        # t-1の単語と重み付けされたエンコーディングで単語を生成
        for t in range(max(dec_lengths)):
            batch_size_t = sum([l > t for l in dec_lengths])

            # コンテキストベクトル, アテンション重み
            context_vector, alpha = self.attention(
                                        encoder_out[:batch_size_t],
                                        h[:batch_size_t])

            # LSTMセル
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            context_vector = gate * context_vector
            h, c = self.decode_step(
                torch.cat([embedded_captions[:batch_size_t, t, :],
                            context_vector], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))

            # 情報保持
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        # Show and tellの出力に合わせる
        encoded_captions = encoded_captions[:, 1:] 
        predictions = pack_padded_sequence(predictions, 
                                            dec_lengths, 
                                            batch_first=True)
        encoded_captions = pack_padded_sequence(encoded_captions, 
                                                dec_lengths, 
                                                batch_first=True)

        return predictions.data, encoded_captions.data, \
               dec_lengths, alphas

    ''' サンプリングによる説明文出力（ビームサーチ無し）
    features:   エンコーダ出力特徴 [バッチサイズ, 埋め込み次元]
    word_to_id: 単語->単語ID辞書
    id_to_word: 単語ID->単語辞書
    '''    
    def sample(self, feature: torch.Tensor, 
               word_to_id: list, id_to_word: list, 
               states=None):
        vocab_size = self.vocab_size

        # 画像を平坦化
        enc_image_size = feature.size(1)
        encoder_dim = feature.size(-1)
        feature = feature.view(1, -1, encoder_dim)
        num_pixels = feature.size(1)
        feature = feature.expand(1, num_pixels, encoder_dim)
        
        # 隠れ状態ベクトル、メモリセルを初期化
        h, c = self.init_hidden_state(feature)

        # <start>を埋め込み
        id_start = word_to_id['<start>']
        prev_words = torch.LongTensor([[id_start]]).to(device) 

        # 予測結果とアテンション重みを保持するためのテンソル
        predictions = []
        alphas = []

        # t-1のデコーダ隠れ状態出力をアテンション重み付けして復号化
        # t-1の単語と重み付けされたエンコーディングで単語を生成
        step = 1
        while True:
            embedded_captions = self.embedding(prev_words).squeeze(1)

            # アテンション重みの計算
            context_vector, alpha = self.attention(feature,h)
            alpha = alpha.view(-1, enc_image_size, enc_image_size)  
            gate = self.sigmoid(self.f_beta(h))
            context_vector = gate * context_vector

            # デコード処理
            h, c = self.decode_step(
                torch.cat([embedded_captions, context_vector], 
                            dim=1), (h, c))

            preds = self.fc(self.dropout(h))
            preds = torch.nn.functional.log_softmax(preds)

            # 単語予測
            prob, predicted = preds.max(1)
            word = id_to_word[predicted.item()]

            # 予測結果とアテンション重みを保存
            predictions.append(predicted)
            alphas.append(alpha)

            # 次のタイムステップへ
            prev_words = torch.LongTensor(
                [predicted.item()]).to(device) 

            # 系列が長くなりすぎたらBreak
            if step > 50:
                break
            step += 1

        return predictions, alphas