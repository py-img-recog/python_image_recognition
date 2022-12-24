import torch
import numpy as np
import copy
from torch import nn
from torchvision import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EncoderCNN(nn.Module):
    ''' Transformer captioningのエンコーダ
    embedding_dim:      埋込みの次元
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


class MultiHeadAttention(nn.Module):
    ''' マルチヘッドアテンション
    embed_dim: 埋め込み層の次元
    num_heads: アテンションヘッドの数
    dropout: Dropout確率
    '''
    def __init__(self, embed_dim: int, 
                 num_heads: int, dropout: float=0.1):
        super().__init__()

        # ヘッド数で特徴量次元を割り切れるか検証
        assert embed_dim % num_heads == 0
        
        # マルチヘッドアテンション構造
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        # モジュール内パラメータ
        self.n_head = num_heads

    ''' マルチヘッドアテンションの順伝播
    query: クエリ [バッチサイズ, ソース系列長, 埋め込み次元]
    key: キー [バッチサイズ, ターゲット系列長, 埋め込み次元]
    value: バリュー [バッチサイズ, ターゲット系列長, 埋め込み次元]
    attn_mask: アテンションマスク
    '''
    def forward(self, query: torch.Tensor, 
                key: torch.Tensor, value: torch.Tensor,
                attn_mask: torch.Tensor=None):

        # N: バッチサイズ
        # S: ソース系列長
        # T: ターゲット系列長
        # E: 埋め込み次元
        # H: ヘッド数
        N, S, E = query.shape
        N, T, E = value.shape
        H = self.n_head

        # マルチヘッドアテンション
        # 複数の線形写像の集合（CNNフィルタのようなもの）

        # (N, H, S, E/H)
        Q = self.query(query).view(N, S, H, E//H).transpose(1,2)
        
        # (N, H, T, E/H)
        K = self.key(key).view(N, T, H, E//H).transpose(1,2)

        # (N, H, T, E/H)
        V = self.value(value).view(N, T, H, E//H).transpose(1,2)
        
        # スケール化内積アテンション
        attn = Q.matmul(K.transpose(2,3)) \
                / torch.sqrt(torch.Tensor([E/H])).to(device)
        
        # マスク処理
        if attn_mask is not None:
            attn = attn.masked_fill(
                attn_mask.to(device) == False, -float('Inf'))

        # ソフトマックス＆ドロップアウト
        A = torch.softmax(attn, dim=3)
        A = self.attn_drop(A)
          
        # バリューとの行列積を算出
        Y = A.matmul(V)
        
        # 出力
        Y = self.proj(Y.transpose(1,2).reshape(N,S,E))

        return Y


class PositionalEncoding(nn.Module):
    '''
    位置埋め込み符号化（Positional encoding）
    embed_dim: 埋込み次元
    dropout: Dropout確率
    max_len: 入力でくる最大系列長
    '''
    def __init__(self, embed_dim: int, 
                 dropout: float=0.1, max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        
        # 位置情報を保持するテンソル（PE）
        pe = torch.zeros(1, max_len, embed_dim)
        col = torch.arange(0, max_len).unsqueeze(1)
        row = 10000**(-torch.arange(0, embed_dim, 2) / embed_dim)
        m = col * row
        pe[:, :, 0::2] = torch.sin(m)
        pe[:, :, 1::2] = torch.cos(m)

        # PEをメモリに保存
        self.register_buffer('pe', pe)

    ''' 位置埋め込み符号化の順伝播
    x: 位置符号を埋め込む対象テンソル [N, S, E]
    '''
    def forward(self, x: torch.Tensor):
        N, S, E = x.shape
        output = torch.empty((N, S, E))
        output = x + self.pe[:, :S, :]  # 位置符号埋込み
        output = self.dropout(output)
        
        return output

class TransformerDecoderLayer(nn.Module):
    ''' Transformerデコーダレイヤ
    input_dim: 特徴量次元
    num_heads: アテンションヘッドの数
    dim_feedforward: FNNの次元
    dropout: Dropout確率
    '''
    def __init__(self, input_dim: int, num_heads: int, 
                 dim_feedforward: int=2048, dropout: int=0.1):
        super().__init__()

        # 単語ベクトルに対するマルチヘッドセルフアテンション
        self.self_attn = MultiHeadAttention(
                            input_dim, num_heads, dropout)
        
        # エンコーダとデコーダ中間出力に対するマルチヘッドアテンション
        self.multihead_attn = MultiHeadAttention(
                            input_dim, num_heads, dropout)
        
        # フィードフォワード層
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        # レイヤ正規化層
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 活性化関数
        self.activation = nn.ReLU()


    ''' Transformerデコーダレイヤの順伝播
    tgt: デコーダへの入力系列 [N, T, W]
    memory: エンコーダレイヤ最終層の系列 [N, S, D]
    tgt_mask: ターゲット系列のマスク [T, T]
    Transformer特徴 [N, T, W]
    '''
    def forward(self, tgt: torch.Tensor, 
                memory: torch.Tensor, tgt_mask: torch.Tensor=None):

        # デコーダ入力に対するマスク付きマルチヘッドセルフアテンション
        tgt2 = self.self_attn(query=tgt, key=tgt, 
                              value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)  # 残差接続
        tgt = self.norm1(tgt)            # レイヤ正規化
        
        # エンコーダとデコーダ中間出力に対するマルチヘッドアテンション
        tgt2 = self.multihead_attn(query=tgt, key=memory,
                                   value=memory)
        tgt = tgt + self.dropout2(tgt2)  # 残差接続
        tgt = self.norm2(tgt)            # レイヤ正規化

        # デコーダ出力
        tgt2 = self.linear2(
                self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)  # 残差接続
        tgt = self.norm3(tgt)            # レイヤ正規化

        return tgt

class TransformerDecoder(nn.Module):
    ''' Transformerデコーダ
    decoder_layer: Transformerデコーダレイヤ
    num_layers: レイヤ数
    '''
    def __init__(self, decoder_layer, num_layers: int):
        super().__init__()

        # num_layersの数だけTransformerデコーダブロックを生成
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    ''' Transformerデコーダの順伝播処理
    tgt: ターゲットとなる出力
    memory: メモリ
    '''
    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output

''' 
レイヤを複製
'''    
def clones(module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class CaptioningTransformer(nn.Module):
    '''
    CaptioningTransformerのコンストラクタ
    '''
    def __init__(self, cfg):
        super().__init__()

        input_dim = cfg.wordvec_dim  # 画像特徴(=単語ベクトル)の次元
        vocab_size = cfg.vocab_size  # 辞書の次元数
        self.vocab_size = vocab_size
        
        self._start = cfg.word_to_id.get("<start>", None)
        self._end = cfg.word_to_id.get("<end>", None)
        self._null = cfg.word_to_id.get("<null>", None)

        # Transformerエンコーダ
        self.visual_projection = nn.Linear(input_dim, 
                                           cfg.wordvec_dim)

        # Transformerデコーダ
        self.embedding = nn.Embedding(vocab_size, 
                                      cfg.wordvec_dim, 
                                      padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(
                                    cfg.wordvec_dim)
        decoder_layer = TransformerDecoderLayer(input_dim=input_dim, 
                                        num_heads=cfg.num_heads)
        self.transformer = TransformerDecoder(decoder_layer, 
                                        num_layers=cfg.num_layers)
        
        # 重みの初期化
        self.apply(self._init_weights)

        # 出力
        self.output = nn.Linear(cfg.wordvec_dim, vocab_size)

    '''
    重みの初期化
    '''
    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    ''' CaptioningTransformerの順伝播処理
    features: 画像特徴ベクトル [N, D]
    captions: 正解キャプション [N, T]
    '''
    def forward(self, features: torch.Tensor,
                captions: torch.Tensor):
        N, T = captions.shape

        # キャプションの埋め込み [N, T] -> [N, T, W]
        caption_embeddings = self.embedding(captions.to(device))

        # 位置符号化
        caption_embeddings = self.positional_encoding(
                                caption_embeddings) 

        # 画像特徴を埋込みと同じ次元に写像
        projected_features = self.visual_projection(
                                features).unsqueeze(1)

        # 未来のキャプションを参照しないようにマスク行列を生成
        tgt_mask = torch.tril(torch.ones(T, T,
                                device=device,
                                dtype=caption_embeddings.dtype))

        # Transformerデコーダでキャプション生成
        # 画像の特徴も入力する
        features = self.transformer(tgt=caption_embeddings,
                                    memory=projected_features,
                                    tgt_mask=tgt_mask)

        # [N, T, W] -> [N, T, V]
        scores = self.output(features)

        return scores

    def sample(self, features: torch.Tensor, 
               max_length: int=30):
        ''' CaptioningTransformerのサンプリング処理
        画像特徴量からキャプショニングを実行
        features: 画像特徴ベクトル [N, D]
        max_length: 最大キャプション長
        '''
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]

            # NULLトークンで出力キャプションを初期化
            captions = self._null * np.ones(
                        (N, max_length), dtype=np.int32)

            # STARTトークンで処理用キャプションを初期化
            partial_caption = self._start * np.ones(N, dtype=np.int32)      
            partial_caption = torch.LongTensor(partial_caption)             
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # トークン予測
                output_logits = self.forward(
                                features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # 確率最大となる単語を選択 [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1).to('cpu')

                # キャプションを更新
                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat(
                                    [partial_caption, word], dim=1)
                
            return captions
