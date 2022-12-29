import random
from typing import Sequence, Dict, Tuple, Union

import torch
from torch.utils.data import Dataset


'''
データセットを分割するための2つの排反なインデックス集合を生成する関数
dataset    : 分割対称のデータセット
ratio      : 1つ目のセットに含めるデータ量の割合
random_seed: 分割結果を不変にするためのシード
'''
def generate_subset(dataset: Dataset, ratio: float,
                    random_seed: int=0):
    # サブセットの大きさを計算
    size = int(len(dataset) * ratio)

    indices = list(range(len(dataset)))

    # 二つのセットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)

    # セット1とセット2のサンプルのインデックスに分割
    indices1, indices2 = indices[:size], indices[size:]

    return indices1, indices2


'''
サンプルからミニバッチを生成するcollate関数
batch     : CocoCaptionsからサンプルした複数の画像とラベルをまとめたもの
word_to_id: 単語->単語ID辞書
'''
def collate_func(batch: Sequence[Tuple[Union[torch.Tensor, str]]],
                 word_to_id: Dict[str, int]):
    imgs, captions = zip(*batch)

    # それぞれのサンプルの5個のキャプションの中から1つを選択してトークナイズ
    captions = [tokenize_caption(
        random.choice(cap), word_to_id) for cap in captions]

    # キャプションの長さが降順になるように並び替え
    batch = zip(imgs, captions)
    batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs)

    lengths = [cap.shape[0] for cap in captions]
    targets = torch.full((len(captions), max(lengths)),
                         word_to_id['<null>'], dtype=torch.int64)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return imgs, targets, lengths


'''
トークナイザ - 文章(caption)を単語IDのリスト(tokens_id)に変換
caption   : 画像キャプション
word_to_id: 単語->単語ID辞書
'''
def tokenize_caption(caption: str, word_to_id: Dict[str, int]):
    tokens = caption.lower().split()
    
    tokens_temp = []    
    # 単語についたピリオド、カンマを削除
    for token in tokens:
        if token == '.' or token == ',':
            continue

        token = token.rstrip('.')
        token = token.rstrip(',')
        
        tokens_temp.append(token)
    
    tokens = tokens_temp        
        
    # 文章(caption)を単語IDのリスト(tokens_id)に変換
    tokens_ext = ['<start>'] + tokens + ['<end>']
    tokens_id = []
    for k in tokens_ext:
        if k in word_to_id:
            tokens_id.append(word_to_id[k])
        else:
            tokens_id.append(word_to_id['<unk>'])
    
    return torch.Tensor(tokens_id)
