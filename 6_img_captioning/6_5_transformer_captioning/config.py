import os
import pickle

class Config(object):
    '''
    ハイパーパラメータ、グローバル変数の設定
    ''' 
    def __init__(self):

        # ハイパーパラメータ（Transformer captioning用）
        self.dropout = 0.30         # dropout確率
        self.num_heads = 4          # ヘッド数
        self.num_layers = 8          # Transformerのレイヤ数
        self.dim_feedforward = 128  # FNNの次元
        self.wordvec_dim = 300     # 単語埋め込み次元（=embedding_dim）

        # ハイパーパラメータ（共通）
        self.embedding_dim = self.wordvec_dim    # 埋め込み層の次元
        self.max_seg_len = 30                    # 最大系列長
        self.learning_rate = 0.001   # 学習率
        self.batch_size = 30        # ミニバッチの数
        self.num_epochs = 30     # エポック
        
        # グローバル変数
        self.fp_train_cap = '/content/drive/MyDrive/data/coco2014/captions_val2014.json'
        self.fp_train_image_dir = 'val2014'
        self.fp_word_to_id = '/content/drive/MyDrive/6_image_captioning/vocab/word_to_id.pkl'
        self.fp_id_to_word = '/content/drive/MyDrive/6_image_captioning/vocab/id_to_word.pkl'
        self.fp_model_dir = '/content/drive/MyDrive/6_image_captioning/model'

        # 辞書（単語→単語ID）の読み込み
        with open(self.fp_word_to_id, 'rb') as f:
            self.word_to_id = pickle.load(f)

        # 辞書（単語ID→単語）の読み込み
        with open(self.fp_id_to_word, 'rb') as f:
            self.id_to_word = pickle.load(f)

        # 辞書サイズ
        self.vocab_size = len(self.word_to_id)

        # モデル出力用のディレクトリ
        if not(os.path.isdir(self.fp_model_dir)):
            os.makedirs(self.fp_model_dir)