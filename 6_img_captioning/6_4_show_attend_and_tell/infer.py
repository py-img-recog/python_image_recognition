import torch
import torchvision.transforms as transforms
import glob
import os
import matplotlib.pyplot as plt
import skimage.transform
import matplotlib.cm as cm
from PIL import Image
from show_attend_and_tell.model import EncoderCNN
from show_attend_and_tell.model import DecoderWithAttention
from show_attend_and_tell.config import Config
from common.util import COCO_loader

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
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # エンコーダモデルの定義
    encoder = EncoderCNN(cfg.enc_image_size, cfg.embedding_dim)
    encoder = encoder.to(device).eval()

    # デコーダモデルの定義
    decoder = DecoderWithAttention(cfg.attention_dim, cfg.embedding_dim, cfg.hidden_dim, cfg.vocab_size)
    decoder = decoder.to(device).eval()

    # モデルの学習済み重みパラメータをロード
    encoder.load_state_dict(torch.load(fp_encoder), strict=False)
    decoder.load_state_dict(torch.load(fp_decoder), strict=False)
    print('エンコーダ: {}'.format(fp_encoder))
    print('デコーダ: {}'.format(fp_decoder))

    for image_file in sorted(glob.glob(os.path.join(fp_infer_image_dir, "*.jpg"))):

        # 画像読み込み
        print("ファイル名: {}".format(os.path.basename(image_file)))
        image = load_image(image_file, transform).to(device)

        # Encoder-decoderによる予測
        with torch.no_grad():

            # encoder
            feature = encoder(image)
            enc_image_size = feature.size(1)
            encoder_dim = feature.size(3)

            # decoder
            predictions, alphas = decoder.sample(feature, cfg.word_to_id, cfg.id_to_word)

        # 可視化
        sampled_caption = []
        word_len = len(predictions)
        image_plt = Image.open(image_file)
        image_plt = image_plt.resize([224, 224], Image.LANCZOS)
        plt.imshow(image_plt)
        plt.axis('off')
        plt.show()
        for t in range(word_len):

            # Attention重みを可視化
            cur_alpha = alphas[t]
            alpha = cur_alpha.to('cpu').numpy()
            alpha = skimage.transform.pyramid_expand(alpha[0, :, :], upscale=16, sigma=8)

            # キャプショニング
            word_id = predictions[t]
            word = cfg.id_to_word[word_id.item()]
            sampled_caption.append(word)

            # タイムステップtの画像をプロット
            plt.imshow(image_plt)
            plt.text(0, 1, '%s' % (word), color='black', backgroundcolor='white', fontsize=12)
            plt.imshow(alpha, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
            plt.show()

            if word == '<end>':
                break
        
        sentence = ' '.join(sampled_caption)
        print ("  {}".format(sentence))

        # 推定結果を書き込み
        gen_sentence_out = image_file[:-4] + "_show_attend_and_tell.txt"
        with open(gen_sentence_out, 'w') as f:
            print("{}".format(sentence), file=f)

