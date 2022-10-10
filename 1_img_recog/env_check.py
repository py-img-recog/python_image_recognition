# PyTorchモジュールのインポート
import torch

if __name__ == '__main__':
    print('Hello world!')

    # Tensorの生成
    x = torch.tensor([1, 2, 3, 4])
    print('CPU', x)

    # TensorをGPUに転送
    x = x.to('cuda')
    print('GPU', x)
