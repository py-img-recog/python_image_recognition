import torch
import torchvision.transforms as transforms
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
from common.util import COCO_loader
from model import EncoderCNN
from model import DecoderRNN
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

    # エンコーダモデルの定義
    encoder = EncoderCNN(cfg.embedding_dim)
    encoder = encoder.to(device).eval()

    # デコーダモデルの定義
    decoder = DecoderRNN(cfg.embedding_dim, cfg.hidden_dim, 
                         cfg.vocab_size, cfg.num_layers, 
                         cfg.max_seg_len)
    decoder = decoder.to(device).eval()

    # モデルの学習済み重みパラメータをロード
    encoder.load_state_dict(torch.load(fp_encoder), strict=False)
    decoder.load_state_dict(torch.load(fp_decoder), strict=False)

    # fp_infer_image_dir内の画像を対象としてキャプショニング実行
    for image_file in sorted(
        glob.glob(os.path.join(fp_infer_image_dir, "*.jpg"))):

        # 画像読み込み
        image = load_image(image_file, transform).to(device)

        # エンコーダ・デコーダモデルによる予測
        with torch.no_grad():
            feature = encoder(image)
            sampled_ids = decoder.sample(feature)

        # 入力画像を表示
        sampled_caption = []
        image_plt = Image.open(image_file)
        image_plt = image_plt.resize([224, 224], Image.LANCZOS)
        plt.imshow(image_plt)
        plt.axis('off')
        plt.show()
        print("入力画像: {}".format(os.path.basename(image_file)))

        # 画像キャプショニングの実行
        sampled_caption = []
        for word_id in sampled_ids:
            word = cfg.id_to_word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        print("出力キャプション: {}".format(sentence))

        # 推定結果を書き込み
        gen_sentence_out = image_file[:-4] + "_show_and_tell.txt"
        with open(gen_sentence_out, 'w') as f:
            print("{}".format(sentence), file=f)

''' 
推論処理
'''
if __name__ == '__main__':
    # 画像キャプショニング推論
    fp_encoder = '/content/drive/MyDrive/6_image_captioning/model/6-3_encoder_best.pth'
    fp_decoder = '/content/drive/MyDrive/6_image_captioning/model/6-3_decoder_best.pth'
    fp_infer_image_dir = '/content/drive/MyDrive/data/image_captioning/'    

    infer(fp_encoder, fp_decoder, fp_infer_image_dir)