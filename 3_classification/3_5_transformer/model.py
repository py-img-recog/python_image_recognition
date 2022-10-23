import copy

import torch
from torch import nn

class SelfAttention(nn.Module):
    ''' Multi-head self Attention機構
    dim_hidden: 入力特徴量の次元
    num_heads:  マルチヘッド注意のヘッド数
    qkv_bias:   クエリなどを生成する全結合でバイアスを使用するかどうか
    '''
    def __init__(self, dim_hidden, num_heads, qkv_bias=False):
        super().__init__()

        # 特徴量を各ヘッドのために分割するので、
        # 特徴量次元をヘッド数で割り切れるか検証
        assert dim_hidden % num_heads == 0

        self.num_heads = num_heads

        # ヘッド毎の特徴次元
        dim_head = dim_hidden // num_heads

        # Softmaxの値が大きくなりすぎないようにするスケール値
        self.scale = dim_head ** -0.5

        # 入力からクエリ、キーおよびバリューを生成
        self.proj_in = nn.Linear(
            dim_hidden, dim_hidden * 3, bias=qkv_bias)

        # 各ヘッドから得られた特徴量をを一つにまとめる
        self.proj_out = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, x):
        # 入力はバッチサイズ * パッチ数 * 特徴量次元
        bs, ns = x.shape[:2]

        qkv = self.proj_in(x)

        # バッチサイズ * パッチ数 * qkv * ヘッド数 * ヘッド次元
        # ↓ permute
        # qkv * バッチサイズ * ヘッド数 * パッチ数 * ヘッド次元
        qkv = qkv.view(
            bs, ns, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # クエリ、キーおよびバリューに分解
        q, k, v = qkv.unbind(0)

        # attnはバッチサイズ * ヘッド数 * パッチ数 * パッチ数
        attn = q.matmul(k.transpose(-2, -1))
        attn = (attn * self.scale).softmax(dim=-1)

        # バッチサイズ * ヘッド数 * パッチ数 * ヘッド次元
        x = attn.matmul(v)

        # flattenによりバッチサイズ * ヘッド数 * パッチ数 * 特徴量次元
        x = x.permute(0, 2, 1, 3).flatten(2)
        x = self.proj_out(x)

        return x

class FNN(nn.Module):
    ''' Feed-forward network
    dim_hidden:      入力特徴量の次元
    dim_feedforward: 中間層の特徴量の次元
    '''
    def __init__(self, dim_hidden, dim_feedforward):
        super().__init__()

        self.linear1 = nn.Linear(dim_hidden, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_hidden)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x

class TransformerEncoderLayer(nn.Module):
    ''' Transformer Encoder layer
    dim_hidden:      入力特徴量の次元
    num_heads:       ヘッド数
    dim_feedforward: 中間特徴量の次元
    '''
    def __init__(self, dim_hidden, num_heads, dim_feedforward):
        super().__init__()

        self.attention = SelfAttention(dim_hidden, num_heads)
        self.fnn = FNN(dim_hidden, dim_feedforward)

        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attention(x) + x
        x = self.norm2(x)
        x = self.fnn(x) + x

        return x

class VisionTransformer(nn.Module):
    ''' Vision Transformer
    num_classes:     識別対象の物体クラス数
    img_size:        入力画像の大きさ(幅=高さであることを想定)
    patch_size:      パッチの大きさ(幅=高さであることを想定)
    dim_hidden:      入力特徴量の次元
    num_heads:       マルチヘッド注意のヘッド数
    dim_feedforward: FNNにおける中間層の特徴量の次元
    num_layers:      Transformerエンコーダの層数
    '''
    def __init__(self, num_classes, img_size, patch_size, dim_hidden,
                 num_heads, dim_feedforward, num_layers):
        super().__init__()

        # 画像をパッチに分解するために、
        # 画像の大きさがパッチの起き差で割り切れるか確認
        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size

        # パッチの行数と列数はともにimg_size // patch_size
        num_patches = (img_size // patch_size) ** 2

        # パッチ特徴量次元はpatch_size * patch_size * 3 (RGBチャネル)
        dim_patch = 3 * patch_size ** 2

        # Transformerエンコーダに乳ロyくする前に
        # パッチ特徴量の次元を変換する全結合
        self.patch_embed = nn.Linear(dim_patch, dim_hidden)

        # 位置埋め込み(パッチ数 + クラス埋め込みの分を用意)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, dim_hidden))

        # クラス埋め込み
        self.class_token = nn.Parameter(
            torch.zeros((1, 1, dim_hidden)))

        # Transformerエンコーダ層
        self.layers = nn.ModuleList([TransformerEncoderLayer(
            dim_hidden, num_heads, dim_feedforward
        ) for _ in range(num_layers)])

        # ロジットを生成する前のレイヤー正規化と全結合
        self.norm = nn.LayerNorm(dim_hidden)
        self.linear = nn.Linear(dim_hidden, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        # 平均0、分散0.02で位置埋め込みとクラス埋め込みを初期化
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, x, return_embed=False):
        bs, c, h, w = x.shape

        # 入力画像の大きさがクラス生成時に指定したimg_sizeと
        # 合致しているか確認
        assert h == self.img_size and w == self.img_size

        # 高さ軸と幅軸をパッチ数 * パッチの大きさに分解し、
        # バッチサイズ * チャネル * パッチの行数 * パッチの大きさ * 
        #                           * パッチの列数 *  パッチの大きさ
        # にする
        x = x.view(bs, c, h // self.patch_size, self.patch_size,
                   w // self.patch_size, self.patch_size)

        # permuteにより
        # バッチサイズ * パッチ行数 * パッチ列数 * チャネル
        #                        * パッチの大きさ *  パッチの大きさ
        x = x.permute(0, 2, 4, 1, 3, 5)

        # パッチを平坦化
        x = x.reshape(
            bs, (h // self.patch_size) * (w // self.patch_size), -1)

        x = self.patch_embed(x)

        # クラス埋め込みをバッチサイズ分用意
        class_token = self.class_token.expand(bs, -1, -1)

        # クラス埋め込みとパッチ特徴を結合
        x = torch.cat((class_token, x), dim=1)

        # 位置埋め込みを加算
        x += self.pos_embed

        # Transformerエンコーダ層を適用
        for layer in self.layers:
            x = layer(x)

        # クラス埋め込みベースの特徴を抽出
        x = x[:, 0]

        x = self.norm(x)

        if return_embed:
            return x

        x = self.linear(x)

        return x

    def get_device(self):
        return self.linear.weight.device

    def copy(self):
        return copy.deepcopy(self)
