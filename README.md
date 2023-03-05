# Pythonで学ぶ画像認識 (機械学習実践シリーズ)

![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-orange)
<a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<img src="https://user-images.githubusercontent.com/55578738/222943631-e4b3e5d4-e856-4c05-aaa9-3ee26ce5e804.jpg" width=35%>

本リポジトリではインプレス社より発売されている[田村 雅人](https://tamtamz.github.io/ja/)・中村 克行 著の機械学習実践シリーズ「**[Pythonで学ぶ画像認識](https://book.impress.co.jp/books/1122101074)**」で扱うソースコードやデータ、学習済みパラメータを管理しています。ソースコードはJupyterノートブックにまとめられており、Google Colabで実行されることを想定しています。ソースコードの解説は書籍内に記載されており、本リポジトリのソースコードは補助教材となっています。

## 書籍の内容

- **第1章 画像認識とは？**
	- 第1節 画像認識の概要
	
	- 第2節 コンピュータによる画像認識の仕組みを理解しよう
	
	- 第3節 実社会で使われている画像認識アプリケーション

	- 第4節 画像認識のための開発環境構築 <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/1_img_recog/1_4_build_env.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>
- **第2章 画像処理の基礎知識**
	- 第1節 画像データを読み込んで表示してみよう <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/2_img_basics/2_1_img_load.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>
	
	- 第2節 画像に平滑化フィルタをかけてみよう <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/2_img_basics/2_2_smoothing_filter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

	- 第3節 畳み込み演算を使った特徴抽出<span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/2_img_basics/2_3_convolution.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

	- 第4節 アテンションを使った特徴抽出<span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/2_img_basics/2_4_attention.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

- **第3章 深層学習を使う準備**
	- 第1節 学習と評価の基礎 <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/3_dnn_prep/3_1_train_eval.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>
	
	- 第2節 深層ニューラルネットワーク
	
- **第4章 画像分類**
	- 第1節 順伝播型ニューラルネットワークによる手法 <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/4_classification/4_1_fnn/4_1_fnn.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>
	
	- 第2節 畳み込みニューラルネットワークによる手法ーResNet18を実装してみよう <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/4_classification/4_2_cnn/4_2_cnn.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

	- 第3節 Transformerによる手法ーVision Transformerを実装してみよう <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/4_classification/4_3_transformer/4_3_transformer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

	- 第4節 精度向上のテクニック <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/4_classification/4_4_technique/4_4_technique.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

- **第5章 物体検出**
	- 第1節 物体検出の基礎 <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/5_object_detection/5_1_object_detection_basics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>
	
	- 第2節 データセットの準備 <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/5_object_detection/5_2_dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

	- 第3節 CNNによる手法ーRetinaNetを実装してみよう <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/5_object_detection/5_3_retinanet/5_3_retinanet.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

	- 第4節 Transformerによる手法ーDETRを実装してみよう <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/5_object_detection/5_4_detr/5_4_detr.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

- **第6章 画像キャプショニング**
	- 第1節 画像キャプショニングの基礎
	
	- 第2節 データセットの準備 <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/6_img_captioning/6_2_dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

	- 第3節 CNN-LSTMによる手法ーShow and tellを実装してみよう <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/6_img_captioning/6_3_show_and_tell/6_3_show_and_tell.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

	- 第4節 アテンション機構による手法ーShow, attend and tellを実装してみよう <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/6_img_captioning/6_4_show_attend_and_tell/6_4_show_attend_and_tell.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

	- 第5節 Transformerによる画像キャプショニングを実装してみよう <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/6_img_captioning/6_5_transformer_captioning/6_5_transformer_captioning.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span>

## 付録

書籍でカバーしきれなかった内容について付録を用意しました。付録はJupyterノートブックで作成されています。

<dl>
<dt><strong>付録A PyTorchの基礎</strong> <span style="float: right"><a target="_blank" href="https://colab.research.google.com/github/py-img-recog/python_image_recognition/blob/main/appendix/a_pytorch.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></span></dt>
<dd style="margin-left: 0;">PyTorchを使う上で最低限必要となる知識について解説しています。PyTorchを始めて使う方は第4章に入る前に本ノートブックを読むことをおすすめします。</dd>
</dl>

## 疑問点・修正点

疑問点や修正点はIssueにて管理しています。不明点などございましたら以下を確認し、解決方法が見つからない場合には新しくIssueを作成してください。

https://github.com/py-img-recog/python_image_recognition/issues