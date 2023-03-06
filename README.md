# Pythonで学ぶ画像認識 (機械学習実践シリーズ)

![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-orange)
<a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<img src="https://user-images.githubusercontent.com/55578738/222943631-e4b3e5d4-e856-4c05-aaa9-3ee26ce5e804.jpg" width=300>

本リポジトリではインプレス社より出版予定の[田村 雅人](https://tamtamz.github.io/ja/)・[中村 克行](https://scholar.google.com/citations?user=ZIxQ5zAAAAAJ&hl=en) 著 機械学習実践シリーズ「**[Pythonで学ぶ画像認識](https://book.impress.co.jp/books/1122101074)**」（3/22発売予定）で扱うソースコードやデータ、学習済みパラメータを管理しています。ソースコードはJupyterノートブックにまとめられており、Google Colabで実行されることを想定しています。ソースコードの解説は書籍内に記載されており、本リポジトリのソースコードは補助教材となっています。

## 書籍の内容

書籍は以下のような構成になります。Jupyterノートブックの補助教材がある節には <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> のバッジをつけています。バッジをクリックすると該当するノートブックをColabで開けます。ただし、この方法でノートブックを開いて画像やラベルデータの読み込みを行う処理を実行した場合、該当するデータがColab上にないためエラーが発生します。ノートブックの処理を実行したい場合には書籍の第1.4節で解説されている環境構築を行って実行してください。

- **第1章 画像認識とは？**
	- 第1節 画像認識の概要
	
	- 第2節 コンピュータによる画像認識の仕組みを理解しよう
	
	- 第3節 実社会で使われている画像認識アプリケーション

	- 第4節 画像認識のための開発環境構築 <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/1_img_recog/1_4_build_env.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **第2章 画像処理の基礎知識**
	- 第1節 画像データを読み込んで表示してみよう <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/2_img_basics/2_1_img_load.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
	
	- 第2節 画像に平滑化フィルタをかけてみよう <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/2_img_basics/2_2_smoothing_filter.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

	- 第3節 畳み込み演算を使った特徴抽出<a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/2_img_basics/2_3_convolution.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

	- 第4節 アテンションを使った特徴抽出<a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/2_img_basics/2_4_attention.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- **第3章 深層学習を使う準備**
	- 第1節 学習と評価の基礎 <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/3_dnn_prep/3_1_train_eval.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
	
	- 第2節 深層ニューラルネットワーク
	
- **第4章 画像分類**
	- 第1節 順伝播型ニューラルネットワークによる手法 <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/4_classification/4_1_fnn/4_1_fnn.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
	
	- 第2節 畳み込みニューラルネットワークによる手法ーResNet18を実装してみよう <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/4_classification/4_2_cnn/4_2_cnn.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

	- 第3節 Transformerによる手法ーVision Transformerを実装してみよう <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/4_classification/4_3_transformer/4_3_transformer.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

	- 第4節 精度向上のテクニック <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/4_classification/4_4_technique/4_4_technique.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- **第5章 物体検出**
	- 第1節 物体検出の基礎 <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/5_object_detection/5_1_object_detection_basics.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
	
	- 第2節 データセットの準備 <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/5_object_detection/5_2_dataset.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

	- 第3節 CNNによる手法ーRetinaNetを実装してみよう <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/5_object_detection/5_3_retinanet/5_3_retinanet.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

	- 第4節 Transformerによる手法ーDETRを実装してみよう <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/5_object_detection/5_4_detr/5_4_detr.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- **第6章 画像キャプショニング**
	- 第1節 画像キャプショニングの基礎
	
	- 第2節 データセットの準備 <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/6_img_captioning/6_2_dataset.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

	- 第3節 CNN-LSTMによる手法ーShow and tellを実装してみよう <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/6_img_captioning/6_3_show_and_tell/6_3_show_and_tell.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

	- 第4節 アテンション機構による手法ーShow, attend and tellを実装してみよう <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/6_img_captioning/6_4_show_attend_and_tell/6_4_show_attend_and_tell.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

	- 第5節 Transformerによる画像キャプショニングを実装してみよう <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/6_img_captioning/6_5_transformer_captioning/6_5_transformer_captioning.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 付録

書籍でカバーしきれなかった内容について付録を用意しました。付録はJupyterノートブックで作成されています。

<dl>
<dt><strong>付録A PyTorchの基礎</strong> <a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/appendix/a_pytorch.ipynb"><img align="right" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></dt>
<dd style="margin-left: 0;">PyTorchを使う上で最低限必要となる知識について解説しています。PyTorchを始めて使う方は第4章に入る前に本ノートブックを読むことをおすすめします。</dd>
</dl>

## 疑問点・修正点

疑問点や修正点はIssueにて管理しています。不明点などございましたら以下を確認し、解決方法が見つからない場合には新しくIssueを作成してください。

https://github.com/py-img-recog/python_image_recognition/issues