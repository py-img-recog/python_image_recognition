import numpy as np
from typing import Callable

import torch
import torchvision


class CocoDetection(torchvision.datasets.CocoDetection):
    '''
    物体検出用COCOデータセット読み込みクラス
    img_directory: 画像ファイルが保存されてるディレクトリへのパス
    anno_file    : アノテーションファイルのパス
    transform    : データ拡張と整形を行うクラスインスタンス
    '''
    def __init__(self, img_directory: str, anno_file: str,
                 transform: Callable=None):
        super().__init__(img_directory, anno_file)

        self.transform = transform

        # カテゴリーIDに欠番があるため、それを埋めてクラスIDを割り当て
        self.classes = []
        # 元々のクラスIDと新しく割り当てたクラスIDを相互に変換する
        # ためのマッピングを保持
        self.coco_to_pred = {}
        self.pred_to_coco = {}
        for i, category_id in enumerate(
                sorted(self.coco.cats.keys())):
            self.classes.append(self.coco.cats[category_id]['name'])
            self.coco_to_pred[category_id] = i
            self.pred_to_coco[i] = category_id

    '''
    データ取得関数
    idx: サンプルを指すインデックス
    '''
    def __getitem__(self, idx: int):
        img, target = super().__getitem__(idx)

        # 親クラスのコンストラクタでself.idsに画像IDが
        # 格納されているのでそれを取得
        img_id = self.ids[idx]

        # 物体の集合を一つの矩形でアノテーションしているものを除外
        target = [obj for obj in target
                  if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # 学習用に当該画像に映る物体のクラスIDと矩形を取得
        # クラスIDはコンストラクタで新規に割り当てたIDに変換
        classes = torch.tensor([self.coco_to_pred[obj['category_id']]
                                for obj in target], dtype=torch.int64)
        boxes = torch.tensor([obj['bbox'] for obj in target],
                             dtype=torch.float32)

        # 矩形が0個のとき、boxes.shape == [0]となってしまうため、
        # 第1軸に4を追加して軸数と第2軸の次元を合わせる
        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4))

        width, height = img.size
        # xmin, ymin, width, height -> xmin, ymin, xmax, ymax
        boxes[:, 2:] += boxes[:, :2]

        # 矩形が画像領域内に収まるように値をクリップ
        boxes[:, ::2] = boxes[:, ::2].clamp(min=0, max=width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)

        # 学習のための正解データを用意
        # クラスIDや矩形など渡すものが多義にわたるため、辞書で用意
        target = {
            'image_id': torch.tensor(img_id, dtype=torch.int64),
            'classes': classes,
            'boxes': boxes,
            'size': torch.tensor((width, height), dtype=torch.int64),
            'orig_size': torch.tensor((width, height),
                                      dtype=torch.int64),
            'orig_img': torch.tensor(np.asarray(img))
        }

        # データ拡張と整形
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    '''
    モデルで予測されたクラスIDからCOCOのクラスIDに変換する関数
    label: 予測されたクラスID
    '''
    def to_coco_label(self, label: int):
        return self.pred_to_coco[label]
