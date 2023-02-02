
import torch
import torch.nn as nn
import torch.nn.functional as file
import torchvision
from collections import OrderedDict

nn_list = [ ('conv1', nn.Conv2d( 3, 15, kernel_size=5)),
            ('conv2', nn.Conv2d(15, 20, kernel_size=7)),
            ('dropout1', nn.Dropout2d()),
            ('affine1', nn.Linear(360 * 512, 50)),
            ('affine2', nn.Linear(50, 10)),
            ('affine3', nn.Linear(10, 2))]

nn_list = OrderedDict(nn_list)

class conv_nn(nn.Module):

    def __init__(self):
        super().__init__()

        self.nn_conv = nn.Sequential(nn_list)
    
    def forward(self, x):

        return self.nn_conv(x)

a = conv_nn()
print(a)
print(a.nn_conv[0])