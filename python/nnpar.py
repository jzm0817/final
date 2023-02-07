import pickle
import platform
import torch.nn as nn
import os
import path
from collections import OrderedDict
from net import trainpar
from net import neuralnetwork

path = path.nnpar_path

# for cur_dir, dirs, files in os.walk(path):
#     index = len(files) 

index = 0

batch_size = 32
epoch = 100
learning_rate = 1e-2
nn_list = [ ('conv1', nn.Conv2d(3, 10, kernel_size=5)),
            ('max_pool1', nn.MaxPool2d(kernel_size=2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(10, 32, kernel_size=5)),
            ('max_pool2', nn.MaxPool2d(kernel_size=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout2d()),
            ('flatten1', nn.Flatten(start_dim=1)),
            ('affine1', nn.Linear(32 * 45 * 45, 50)),
            ('affine2', nn.Linear(50, 10)),
            ('affine3', nn.Linear(10, 4)),
            ]

nn_list = OrderedDict(nn_list)

trainpar = trainpar(batch_size, learning_rate, epoch, nn_list)

output_hal = open(path + '/' + "par_" + str(index) + ".pkl", 'wb')
output_hal.write(pickle.dumps(trainpar))
output_hal.close()
