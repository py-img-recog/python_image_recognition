import os
import pickle
import numpy as np
import datetime
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from torch.utils.data.dataset import Subset
from show_and_tell.model import EncoderCNN
from show_and_tell.model import DecoderRNN
from show_and_tell.config import Config
from common.util import COCO_loader

'''
Show and Tellの学習
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
    decoder = DecoderRNN(cfg.embedding_dim, 
                         cfg.hidden_dim, 
                         cfg.vocab_size, 
                         cfg.num_layers).to(device)
    
    # 損失関数の定義
    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.word_to_id.get("<null>", None))
    
    # 最適化手法の定義
    params = list(decoder.parameters()) \
             + list(encoder.linear.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)

    # 学習率スケジューラの定義
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[20,50], gamma=0.1)
    
    # 学習経過の書き込み
    now = datetime.datetime.now()
    fp_train_loss_out = '{}/6-1_train_loss_{}.csv'\
        .format(cfg.fp_model_dir, now.strftime('%Y%m%d_%H%M%S'))
    fp_val_loss_out = '{}/6-1_val_loss_{}.csv'\
        .format(cfg.fp_model_dir, now.strftime('%Y%m%d_%H%M%S'))

    # 学習
    print("学習開始")
    for epoch in range(cfg.num_epochs):
        with tqdm(train_loader) as pbar:
            pbar.set_description("[Train epoch %d]" % (epoch + 1))
            train_losses = []
            for i, (images, captions,lengths) in enumerate(pbar):

                # 学習モードに設定
                encoder.train()
                decoder.train()

                # ミニバッチを設定
                images = images.to(device)
                captions = captions.to(device)

                # エンコーダ-デコーダモデル
                features = encoder(images)
                outputs = decoder(features, captions, lengths)

                # ロスの計算
                targets = pack_padded_sequence(captions, 
                                               lengths, 
                                               batch_first=True)[0]
                loss = criterion(outputs, targets)

                # backward
                optimizer.zero_grad()
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
                images = images.to(device)
                captions = captions.to(device)

                # エンコーダ-デコーダモデル
                features = encoder(images)
                outputs = decoder(features, captions, lengths)

                # ロスの計算
                targets = pack_padded_sequence(captions, 
                                               lengths, 
                                               batch_first=True)[0]
                val_loss = criterion(outputs, targets)
                val_losses.append(val_loss.item())

                # Validation Lossをログに書き込み
                with open(fp_val_loss_out, 'a') as f:
                    print("{},{}".format(epoch, 
                                         val_loss.item()), 
                                         file=f)

        # Loss 表示
        print("Validation loss: {}".format(np.average(val_losses)))

        # モデル保存
        if (epoch % 10) == 0 or ((epoch+1) == cfg.num_epochs):

            # エンコーダモデルを保存
            fp_encoder = '{}/6-1_encoder-epoch{:02d}-loss{:.0f}.pth'.format(cfg.fp_model_dir, epoch+1, val_loss.item()*100)
            torch.save(encoder.to('cpu').state_dict(), fp_encoder)
            encoder.to(device)

            # デコーダモデルを保存
            fp_decoder = '{}/6-1_decoder-epoch{:02d}-loss{:.0f}.pth'.format(cfg.fp_model_dir, epoch+1, val_loss.item()*100)
            torch.save(decoder.to('cpu').state_dict(), fp_decoder)
            decoder.to(device)
    
    print("学習終了")

''' 
学習処理
'''
if __name__ == '__main__':
    train()
