
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict

ch_in = 32
reduction = 16

se_list = [
    ("affine1", nn.Linear(ch_in, ch_in // reduction, bias=False)),
    ("relu1", nn.ReLU(inplace=True)),
    ("affine2", nn.Linear(ch_in // reduction, ch_in, bias = False)),
    ("sigmoid", nn.Sigmoid()) 
]


class se_block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(se_list)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


se_list = OrderedDict(se_list)

se = se_block(ch_in)

nn_list = [ ('conv1', nn.Conv2d(3, 10, kernel_size=5)),
            ('max_pool1', nn.MaxPool2d(kernel_size=2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(10, 32, kernel_size=5)),
            ('max_pool2', nn.MaxPool2d(kernel_size=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout2d()),
            ('se_block', se),
            ('flatten1', nn.Flatten(start_dim=1)),
            ('affine1', nn.Linear(32 * 45 * 45, 50)),
            ('affine2', nn.Linear(50, 10)),
            ('affine3', nn.Linear(10, 4)),
            ]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_net = nn.Sequential(nn_list)

    def forward(self, x):
        return self.conv_net(x)

nn_list = OrderedDict(nn_list)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)
 
summary(model, (3, 96 * 2, 96 * 2))


