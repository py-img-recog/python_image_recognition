import pickle
from pycocotools.coco import COCO
from collections import Counter

# データの保存先
fp_train_caption = '/content/drive/MyDrive/data/coco2014/captions_val2014.json'
fp_word_to_id = '/content/drive/MyDrive/6_image_captioning/vocab/word_to_id.pkl'
fp_id_to_word = '/content/drive/MyDrive/6_image_captioning/vocab/id_to_word.pkl'

# キャプションを読み込み
coco = COCO(fp_train_caption)
anns_keys = coco.anns.keys()

# 単語ーID対応表の作成
coco_token = []
for key in anns_keys:
    caption = coco.anns[key]['caption']
    tokens = caption.lower().split()
    coco_token.extend(tokens)

# ピリオド、カンマを削除
table = str.maketrans({"." : "",
                       "," : ""})
for k in range(len(coco_token)):
    coco_token[k] = coco_token[k].translate(table)

# 単語ヒストグラムを作成
freq = Counter(coco_token)

# 3回以上出現する単語を限定して辞書を作成
vocab = []
common = freq.most_common()
for t,c in common:
    if c >= 3:
        vocab.append(t)
sorted(vocab)

# 特殊トークンの追加
vocab.append('<start>') # 文書の始まりを表すトークンを追加
vocab.append('<end>') # 文書の終わりを表すトークンを追加
vocab.append('<unk>') # 辞書内に無い単語を表すトークンを追加
vocab.append('<null>') # 系列長を揃えるためのトークンを追加

# 単語ー単語ID対応表の作成
word_to_id = {t:i for i,t in enumerate(vocab)}
id_to_word = {i:t for i,t in enumerate(vocab)}

# ファイル出力
with open(fp_word_to_id, 'wb') as f:
    pickle.dump(word_to_id, f)
with open(fp_id_to_word, 'wb') as f:
    pickle.dump(id_to_word, f)

print('単語数: ' + str(len(word_to_id)))