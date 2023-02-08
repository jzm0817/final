import pickle
import platform
import torch
import torch.nn as nn
import os
import path
from collections import OrderedDict
import net
import argparse

import hiddenlayer as h

def default_argument_parser():
    parser = argparse.ArgumentParser(description="pytorch-learning")
    parser.add_argument('--show', action="store_true", help="show parameters")
    parser.add_argument('--save', action="store_true", help="save .pkl")
    # parser.add_argument('--nn', '--nn-model', default=pic_size)
    return parser

args = default_argument_parser().parse_args()

if args.show:
    show_flag = True
else:
    show_flag = False 

if args.save:
    save_flag = True
else:
    save_flag = False 


path = path.nnpar_path

# for cur_dir, dirs, files in os.walk(path):
#     index = len(files) 

index = 3

batch_size = 64
epoch = 100
learning_rate = 1e-4
# nn_list = [ ('conv1', nn.Conv2d(3, 10, kernel_size=5)),
#             ('max_pool1', nn.MaxPool2d(kernel_size=2)),
#             ('relu1', nn.ReLU(inplace=True)),
#             ('conv2', nn.Conv2d(10, 32, kernel_size=5)),
#             ('max_pool2', nn.MaxPool2d(kernel_size=2)),
#             ('relu2', nn.ReLU(inplace=True)),
#             # ('dropout1', nn.Dropout2d()),
#             ('flatten1', nn.Flatten(start_dim=1)),
#             ('affine1', nn.Linear(32 * 45 * 45, 50)),
#             ('affine2', nn.Linear(50, 10)),
#             ('affine3', nn.Linear(10, 4)),
#             ]


# nn_list = OrderedDict(nn_list)
# model = net.neuralnetwork(nn_list)
model = net.resnet(net.residual_block, [2,2,2,2])

trainpar = net.trainpar(batch_size, learning_rate, epoch, model)

if show_flag:
    print(f'index:{index}')
    print(f'epoch:{epoch}')
    print(f'batch_size:{batch_size}')
    print(f'learning_rate:{learning_rate}')
    print(f'model:{model}')

if save_flag:
    output_hal = open(path + '/' + "par_" + str(index) + ".pkl", 'wb')
    output_hal.write(pickle.dumps(trainpar))
    output_hal.close()
    print(f'save file:{path}/par{str(index)}.pkl')


# vis_graph = h.build_graph(model, torch.zeros(batch_size, 3, 96 * 2, 96 * 2))   # 获取绘制图像的对象
# vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
# vis_graph.save("demo1", format='jpg') 