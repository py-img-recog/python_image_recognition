import os
import pickle
import numpy as np
import datetime
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from torch.utils.data.dataset import Subset
from timm.scheduler import CosineLRScheduler
from model import EncoderCNN
from model import CaptioningTransformer
from config import Config

import sys
sys.path.append('..')
from common.util import COCO_loader

'''
Transformer Captioningの学習
'''
def train():
    # GPUの設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # ハイパーパラメータの設定
    cfg = Config()

    # 学習データの読み込み
    train_loader, valid_loader = COCO_loader(cfg.batch_size, 
                                    cfg.word_to_id, 
                                    cfg.fp_train_cap, 
                                    cfg.fp_train_image_dir)

    # モデルの定義
    encoder = EncoderCNN(cfg.embedding_dim).to(device)
    transformer = CaptioningTransformer(cfg).to(device)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.word_to_id.get("<null>", None))

    # 最適化手法の定義
    params = list(transformer.parameters()) + \
                list(encoder.linear.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)

    # WarmupとCosine Decayを行うスケジューラを利用
    scheduler = CosineLRScheduler(optimizer, t_initial=cfg.num_epochs, 
                                    lr_min=1e-4, warmup_t=20, 
                                    warmup_lr_init=5e-5, warmup_prefix=True)

    # 学習経過の書き込み
    now = datetime.datetime.now()
    fp_train_loss_out = '{}/6-5_train_loss_{}.csv'\
        .format(cfg.fp_model_dir, now.strftime('%Y%m%d_%H%M%S'))
    fp_val_loss_out = '{}/6-5_val_loss_{}.csv'\
        .format(cfg.fp_model_dir, now.strftime('%Y%m%d_%H%M%S'))

    # 学習
    print("学習開始")
    val_loss_best = float('inf')
    for epoch in range(cfg.num_epochs):
        with tqdm(train_loader) as pbar:
            pbar.set_description("[Train epoch %d]" % (epoch + 1))
            train_losses = []
            for i, (images, captions, lengths) in enumerate(pbar):

                # 学習モード
                encoder.train()
                transformer.train()

                # ミニバッチを設定
                images, captions = images.to(device), captions.to(device)
                t_captions_in = captions[:, :-1]  # <end>を除外
                t_captions_out = captions[:, 1:] # <start>を除外
                t_mask = t_captions_out!=transformer._null

                # Forward
                features = encoder(images)
                logits = transformer(features, t_captions_in)
                loss = criterion(logits.reshape(-1, logits.shape[-1]),
                                 t_captions_out.reshape(-1))

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Training Lossをログに書き込み
                train_losses.append(loss.item())
                with open(fp_train_loss_out, 'a') as f:
                    print("{},{}".format(epoch, loss.item()), file=f)

        # 学習率 表示
        print("LR: {}".format(scheduler.get_epoch_values(epoch)))

        # Loss 表示
        print("Training loss: {}".format(np.average(train_losses)))

        # validation
        with tqdm(valid_loader) as pbar:
            pbar.set_description("[Validation %d]" % (epoch + 1))
            val_losses = []
            for j, (images, captions,lengths) in enumerate(pbar):

                # 評価モード
                encoder.eval()
                transformer.eval()

                # ミニバッチを設定
                images, captions = images.to(device), captions.to(device)
                t_captions_in = captions[:, :-1]  # <end>を除外
                t_captions_out = captions[:, 1:] # <start>を除外
                t_mask = t_captions_out!=transformer._null

                # Forward
                with torch.no_grad():
                    features = encoder(images)
                    logits = transformer(features, t_captions_in)
                    val_loss = criterion(logits.reshape(-1, logits.shape[-1]),
                        t_captions_out.reshape(-1))

                    pred_captions = transformer.sample(features, max_length=30)

                val_losses.append(val_loss.item())

                # Validation Lossをログに書き込み
                with open(fp_val_loss_out, 'a') as f:
                    print("{},{}".format(epoch, val_loss.item()), file=f)
                    
        # Loss 表示
        val_loss = np.average(val_losses)
        print("Validation loss: {}".format(val_loss))

        # より良い検証結果が得られた場合、モデルを保存
        if val_loss < val_loss_best:
            val_loss_best = val_loss

            # エンコーダモデルを保存
            fp_encoder = '{}/6-5_encoder_best.pth'.format(cfg.fp_model_dir)
            torch.save(encoder.to('cpu').state_dict(), fp_encoder)
            encoder.to(device)

            # transformerモデルを保存
            fp_decoder = '{}/6-5_decoder_best.pth'.format(cfg.fp_model_dir)
            torch.save(transformer.to('cpu').state_dict(), fp_decoder)
            transformer.to(device)
    
    print("学習終了")

if __name__ == '__main__':
    train()