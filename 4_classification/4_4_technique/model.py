import torch
from torch import nn


class BasicBlock(nn.Module):
    ''' 
    ResNet18における残差ブロック
    in_channels : 入力チャネル数
    out_channels: 出力チャネル数
    stride      : 畳み込み層のストライド
    '''
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int=1):
        super().__init__()

        ''''' 残差接続 '''''
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        ''''''''''''''''''''

        # strideが1より大きいときにスキップ接続と残差接続の高さと幅を
        # 合わせるため、別途畳み込み演算を用意
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    '''
    順伝播関数
    x: 入力, [バッチサイズ, 入力チャネル数, 高さ, 幅]
    '''
    def forward(self, x: torch.Tensor):
        ''''' 残差接続 '''''
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        ''''''''''''''''''''

        if self.downsample is not None:
            x = self.downsample(x)

        # 残差写像と恒等写像の要素毎の和を計算
        out += x

        out = self.relu(out)

        return out
