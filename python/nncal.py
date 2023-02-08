
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict

import path
import pickle
import net

# index = 0

# with open(path.nnpar_path + '/' + "par_" + str(index) + ".pkl", 'rb') as f:
#     par = pickle.loads(f.read())




class residual_block(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, downsample=None):
        super().__init__()
        self.conv1=nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.relu=nn.ReLU(inplace=True)
        
        self.conv2=nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(ch_out)
        self.downsample=downsample

    
    def forward(self, x):
        residual=x
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        
        if self.downsample:
            residual=self.downsample(x)
        
        out+=residual
        out=self.relu(out)

        return out

class resnet(nn.Module):
    def __init__(self, block, layers, num_classes=4):
        super().__init__()
        self.ch_in = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * 6 * 6, num_classes)
    
    def make_layer(self, block, ch_out, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.ch_in != ch_out):
            downsample = nn.Sequential(
            nn.Conv2d(self.ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out)
                                     )
        layers = []
        layers.append(block(self.ch_in, ch_out, stride, downsample))
        self.ch_in = ch_out
        for i in range(1, blocks):
            layers.append(block(ch_out, ch_out))
        return nn.Sequential(*layers)# add all of the residual block
            
    
    def forward(self,x):
        out = self.conv(x) 
        out = self.bn(out) 
        out = self.relu(out) 
        out = self.layer1(out) 
        out = self.layer2(out) 
        out = self.layer3(out) 
        out = self.avg_pool(out) 
        out = out.view(out.size(0), -1) 
        out = self.fc(out) 
        
        return out 

ch_in = 3
ch_out = 16
stride = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# nn_list = [
#     ('0', nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False)),
#         ('1', nn.BatchNorm2d(ch_out)),
#         ('2', nn.ReLU(inplace=True)),
        
#         ('3', nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)),
#         ('4', nn.BatchNorm2d(ch_out))
#         ]


# nn_list = OrderedDict(nn_list)

# model = net.neuralnetwork(nn_list).to(device)

 
# res_bk = residual_block()
model=resnet(residual_block, [2,2,2,2]).to(device)

summary(model, (3, 96 * 2, 96 * 2))


