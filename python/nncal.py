
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict
import argparse

import path
import pickle
import net


def default_argument_parser():
    parser = argparse.ArgumentParser(description="pytorch-learning")
    parser.add_argument('--show', action="store_true", help="show parameters")
    parser.add_argument('--id', type=int, default=0)
    return parser


args = default_argument_parser().parse_args()

index = args.id

if args.show:
    show_flag = True
else:
    show_flag = False 


with open(path.nnpar_path + '/' + "par_" + str(index) + ".pkl", 'rb') as f:
    par = pickle.loads(f.read())
print('------------------------------------')
print(f'load par_{index}.pkl')
print('------------------------------------')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

model = par.ann
model = model.to(device)

if show_flag:
    print(f'pic_size:{par.pic_size}')
    print(f'pic_list:{par.pic_list}')
    print(f'pic_enhance_list:{par.pic_enhance_list}')
    print(f'index:{index}')
    print(f'epoch:{par.epoch}')
    print(f'batch_size:{par.bs}')
    print(f'learning_rate:{par.lr}')
    # print(f'model:{par.ann}')
    summary(model, (3, 96 * 2, 96 * 2))


