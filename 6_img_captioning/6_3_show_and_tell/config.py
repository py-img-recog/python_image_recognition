import os
import pickle

class Config(object):
    '''
    ハイパーパラメータ、グローバル変数の設定
    '''    
    def __init__(self):

        # ハイパーパラメータ
        self.embedding_dim = 300  # 埋め込み層の次元
        self.hidden_dim = 128      # LSTM隠れ層の次元
        self.num_layers = 2         # LSTM階層の数
        self.max_seg_len = 30     # 最大系列長
        self.learning_rate = 0.001  # 学習率
        self.dropout = 0.30         # dropout確率
        self.batch_size = 30        # ミニバッチ数
        self.num_epochs = 100     # エポック数
        
        # グローバル変数
        self.fp_train_cap = '../data/coco2014/captions_val2014.json'
        self.fp_train_image_dir = '../data/coco2014/val2014'
        self.fp_infer_image_dir = '../data/coco2014/infer_test'
        self.fp_word_to_id = 'vocab/word_to_id.pkl'
        self.fp_id_to_word = 'vocab/id_to_word.pkl'
        self.fp_model_dir = 'model'

        # 辞書（単語→単語ID）の読み込み
        with open(self.fp_word_to_id, 'rb') as f:
            self.word_to_id = pickle.load(f)

        # 辞書（単語ID→単語）の読み込み
        with open(self.fp_id_to_word, 'rb') as f:
            self.id_to_word = pickle.load(f)

        # 辞書サイズを保存
        self.vocab_size = len(self.word_to_id)
        
        # モデル出力用のディレクトリを作成
        if not(os.path.isdir(self.fp_model_dir)):
            os.makedirs(self.fp_model_dir)