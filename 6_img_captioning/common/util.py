import pickle
import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
from pycocotools.coco import COCO

'''
COCOデータセットローダ
batch_size:         バッチサイズ
word_to_id:         単語->単語ID辞書
fp_train_caption:   学習用のキャプション
fp_train_image_dir: 学習画像のパス
'''
def COCO_loader(batch_size: int, word_to_id: list, 
                fp_train_caption: str, 
                fp_train_image_dir: str):

    ''' トークナイザ
    文章(caption)を単語IDのリスト(tokens_id)に変換
    caption: 画像キャプション [バッチサイズ, 系列長]
    '''
    def tokenize_caption(caption: torch.Tensor):
        # 単語についたピリオド、カンマを削除
        tokens = caption.lower().split()
        tokens_temp = []
        for t in tokens:
            if t.endswith('.') and t != '.':
                tokens_temp.append(t.replace('.', ''))
            elif t.endswith(',') and t != ',':
                tokens_temp.append(t.replace(',', ''))
            elif t == '.' or t == ',':
                continue
            else:
                tokens_temp.append(t)
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

    '''
    COCOデータセットからデータを取り出すためのcollate関数
    '''
    def cap_collate_fn(data):
        images, captions = zip(*data)
        captions = [tokenize_caption(cap[random.randrange(len(cap))]) for cap in captions]
        
        data = zip(images, captions)
        data = sorted(data, key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)
        images = torch.stack(images, 0)

        lengths = [len(c) for c in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        targets[:] = word_to_id['<null>']   # nullでパディング
        for i,c in enumerate(captions):
            end = lengths[i]
            targets[i,:end] = c[:end]
        return images, targets, lengths
 
    # 画像のtransformsを定義
    crop_size = (224,224)             # CNN入力画像サイズ
    in_mean = (0.485, 0.456, 0.406)   # ImageNetの平均値
    in_std = (0.229, 0.224, 0.225)    # ImageNetの標準偏差
    trans = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(in_mean, in_std) 
    ])

    # COCOデータロードの定義
    train_val_set = dset.CocoCaptions(root=fp_train_image_dir, 
                                        annFile=fp_train_caption, 
                                        transform=trans)
            
    # データサブセットを取得するサンプラーの定義
    # 学習データ70%、評価データ30%に分割
    n_samples = len(train_val_set)
    indices = list(range(n_samples))
    tr_split = int(0.7 * n_samples)      
    train_idx, val_idx = indices[:tr_split], indices[tr_split:]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Dataloaderを生成
    train_loader = torch.utils.data.DataLoader(
                        train_val_set, 
                        batch_size=batch_size, 
                        num_workers=4, 
                        sampler=train_sampler,
                        collate_fn=cap_collate_fn)

    val_loader = torch.utils.data.DataLoader(
                        train_val_set, 
                        batch_size=batch_size, 
                        num_workers=4, 
                        sampler=val_sampler,
                        collate_fn=cap_collate_fn)
                                            
    return train_loader, val_loader