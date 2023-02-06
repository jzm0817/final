
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
            ('flatten1', nn.Flatten(start_dim=1)),
            ('affine1', nn.Linear(20 * 45 * 45, 50)),
            ('affine2', nn.Linear(50, 10)),
            ('affine3', nn.Linear(10, 4)),
            ]


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 15, kernel_size=5)
#         self.pool1 = nn.MaxPool2d(2)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(15, 20, kernel_size=7)
#         self.conv2_drop = nn.Dropout2d()
#         self.pool2 = nn.MaxPool2d(2)
#         self.relu2 = nn.ReLU(inplace=True)
#         # self.fc1 = nn.Linear(81000, 50)
#         # self.fc2 = nn.Linear(50, 10)
 
#     def forward(self, x):
#         # x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.conv2_drop(x)
#         x = self.relu2(x)
#         # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         # x = x.view(-1, 81000)
#         # x = F.relu(self.fc1(x))
#         # x = F.dropout(x, training=self.training)
#         # x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

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


