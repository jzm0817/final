
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict


nn_list = [ ('conv1', nn.Conv2d(3, 10, kernel_size=5)),
            ('max_pool1', nn.MaxPool2d(kernel_size=2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(10, 20, kernel_size=5)),
            ('max_pool2', nn.MaxPool2d(kernel_size=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout2d()),
            # ('flatten1', nn.Flatten(start_dim=1)),
            # ('affine1', nn.Linear(20 * 45 * 45, 50)),
            # ('affine2', nn.Linear(50, 10)),
            # ('affine3', nn.Linear(10, 4)),
            ]

nn_list = OrderedDict(nn_list)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_net = nn.Sequential(nn_list)

    def forward(self, x):
        return self.conv_net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)
 
summary(model, (3, 96 * 2, 96 * 2))


