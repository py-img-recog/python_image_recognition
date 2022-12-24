import datetime
import torch
import torchvision.transforms as transforms
import pickle
import glob
import sys
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from model import EncoderCNN
from model import CaptioningTransformer
from config import Config

''' 画像読み込み
image_file:   画像ファイル
transform:    画像変換
'''
def load_image(image_file: str, transform=None):
    image = Image.open(image_file)
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

''' 
画像キャプショニングの推論
'''
def infer(fp_encoder: str, fp_decoder: str, fp_infer_image_dir: str):
 
    # GPUを利用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running in %s." % device)

    # パラメータ設定
    cfg = Config()
    
    # 画像の正規化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])
    # モデルの定義
    encoder = EncoderCNN(cfg.embedding_dim).to(device)
    encoder.eval()
    transformer = CaptioningTransformer(cfg).to(device)
    transformer.eval()

    # モデルの学習済み重みパラメータをロード
    encoder.load_state_dict(torch.load(fp_encoder), strict=False)
    transformer.load_state_dict(torch.load(fp_decoder), strict=False)
    print('エンコーダ: {}'.format(fp_encoder))
    print('デコーダ: {}'.format(fp_decoder))

    for image_file in sorted(glob.glob(os.path.join(fp_infer_image_dir, "*.jpg"))):

        # 画像読み込み
        print("ファイル名: {}".format(os.path.basename(image_file)))
        image = load_image(image_file, transform).to(device)

        # キャプション予測
        with torch.no_grad():

            feature = encoder(image)
            pred_captions = transformer.sample(feature, max_length=30)

        # 可視化
        sampled_caption = []
        word_len = pred_captions.shape[1]
        image_plt = Image.open(image_file)
        image_plt = image_plt.resize([224, 224], Image.LANCZOS)
        plt.imshow(image_plt)
        plt.axis('off')
        plt.show()

        for t in range(word_len):

            # キャプショニング
            word_id = pred_captions[:,t]
            word = cfg.id_to_word[word_id.item()]
            sampled_caption.append(word)

            if word == '<end>':
                break
        
        sentence = ' '.join(sampled_caption)
        print ("  {}".format(sentence))

        # 推定結果を書き込み
        gen_sentence_out = image_file[:-4] + "_transformer.txt"
        with open(gen_sentence_out, 'w') as f:
            print("{}".format(sentence), file=f)

''' 
推論処理
'''
if __name__ == '__main__':
    # 画像キャプショニング推論
    fp_encoder = '/content/drive/MyDrive/6_image_captioning/model/6-5_encoder_best.pth'
    fp_decoder = '/content/drive/MyDrive/6_image_captioning/model/6-5_decoder_best.pth'
    fp_infer_image_dir = '/content/drive/MyDrive/data/image_captioning/'    

    infer(fp_encoder, fp_decoder, fp_infer_image_dir)