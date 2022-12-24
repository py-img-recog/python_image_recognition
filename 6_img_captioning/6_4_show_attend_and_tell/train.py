import os
import pickle
import numpy as np
import datetime
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from torch.utils.data.dataset import Subset
from model import EncoderCNN
from model import DecoderWithAttention
from config import Config

import sys
sys.path.append('..')
from common.util import COCO_loader

'''
Show, attend and Tellの学習
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
    encoder = EncoderCNN(cfg.enc_image_size, 
                         cfg.embedding_dim).to(device)
    decoder = DecoderWithAttention(cfg.attention_dim, 
                                    cfg.embedding_dim, 
                                    cfg.hidden_dim, 
                                    cfg.vocab_size).to(device)
    
    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # 最適化手法の定義
    params = list(decoder.parameters()) + \
             list(encoder.adaptive_pool.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)

    # 学習経過の書き込み
    now = datetime.datetime.now()
    fp_train_loss_out = '{}/6-4_train_loss_{}.csv'\
        .format(cfg.fp_model_dir, now.strftime('%Y%m%d_%H%M%S'))
    fp_val_loss_out = '{}/6-4_val_loss_{}.csv'\
        .format(cfg.fp_model_dir, now.strftime('%Y%m%d_%H%M%S'))

    # 学習
    print("学習開始")
    val_loss_best = float('inf')
    for epoch in range(cfg.num_epochs):
        with tqdm(train_loader) as pbar:
            pbar.set_description("[Train epoch %d]" % (epoch + 1))
            train_losses = []
            for i, (images, captions,lengths) in enumerate(pbar):

                # 学習モード
                encoder.train()
                decoder.train()

                # ミニバッチを設定
                images, captions = \
                    images.to(device), captions.to(device)
                targets = pack_padded_sequence(captions, 
                                               lengths, 
                                               batch_first=True)[0]                
                optimizer.zero_grad()

                # Forward
                features = encoder(images)
                outputs, targets, decode_lengths, alphas = \
                    decoder(features, captions, lengths)
                loss = criterion(outputs, targets)

                # backward
                loss.backward()
                optimizer.step()

                # Training Lossをログに書き込み
                train_losses.append(loss.item())
                with open(fp_train_loss_out, 'a') as f:
                    print("{},{}".format(epoch, loss.item()), file=f)

        # Loss 表示
        print("Training loss: {}".format(np.average(train_losses)))

        # validation
        with tqdm(valid_loader) as pbar:
            pbar.set_description("[Validation %d]" % (epoch + 1))
            val_losses = []
            for j, (images, captions,lengths) in enumerate(pbar):

                # 評価モード
                encoder.eval()
                decoder.eval()

                # ミニバッチを設定
                images, captions = \
                    images.to(device), captions.to(device)
                targets = pack_padded_sequence(captions, 
                                               lengths, 
                                               batch_first=True)[0]

                features = encoder(images)
                outputs, targets, decode_lengths, alphas = \
                    decoder(features, captions, lengths)
                val_loss = criterion(outputs, targets)
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
            fp_encoder = '{}/6-4_encoder_best.pth'.format(cfg.fp_model_dir)
            torch.save(encoder.to('cpu').state_dict(), fp_encoder)
            encoder.to(device)

            # デコーダモデルを保存
            fp_decoder = '{}/6-4_decoder_best.pth'.format(cfg.fp_model_dir)
            torch.save(decoder.to('cpu').state_dict(), fp_decoder)
            decoder.to(device)
    
    print("学習終了")

if __name__ == '__main__':
    train()