
import torch
import torch.nn as nn
import torch.nn.functional as file
import torchvision
from collections import OrderedDict

nn_list = [ ('conv1', nn.Conv2d(3, 10, kernel_size=5)),
            ('max_pool1', nn.MaxPool2d(kernel_size=2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(10, 20, kernel_size=5)),
            ('max_pool2', nn.MaxPool2d(kernel_size=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout2d()),
            ('flatten1', nn.Flatten(start_dim=1)),
            ('affine1', nn.Linear(20 * 93 * 93, 50)),
            ('affine2', nn.Linear(50, 10)),
            ('affine3', nn.Linear(10, 4)),
            ]


nn_list = OrderedDict(nn_list)

class conv_nn(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.nn_conv = nn.Sequential(nn_list)
    
    def forward(self, x):
        # x = self.flatten(x)
        logits = self.nn_conv(x)
        return logits

# a = conv_nn()
# print(a)
# print(a.nn_conv[0])