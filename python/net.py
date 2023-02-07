
import path
import par
from plotcm import plot_confusion_matrix


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as file
import torchvision
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



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


nn_list = OrderedDict(nn_list)

class convnn(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.nn_conv = nn.Sequential(nn_list)
    
    def forward(self, x):
        # x = self.flatten(x)
        logits = self.nn_conv(x)
        return logits



def train(model, data_set_training, optimizer, loss_fn, epoch, device):
    model.train()

    loss_total = 0

    for _, data in enumerate(data_set_training):
        if isinstance(data, list):
            image = data[0].type(torch.FloatTensor).to(device)
            label = data[1].to(device)
        elif isinstance(data, dict):
            image = data['image'].type(torch.FloatTensor).to(device)
            label = data['label'].to(device)
        else:
            print(type(data))
            raise TypeError
        
        optimizer.zero_grad()
        # print(f'image.shape:{image.shape}')
        output = model(image)

        loss = loss_fn(output, label)
        loss_total+=loss.item()

        loss.backward()
        optimizer.step()
    
    print(f'{round(loss_total,2)} in epoch {epoch}')
    return loss_total


def test(model, data_set_test, device, bs, trained_name):
    model.eval()

    correct = 0
    real_label = []
    pred_label = []
    bs = 0

    for ii, data in enumerate(data_set_test):

        if isinstance(data, list):
            image = data[0].type(torch.FloatTensor).to(device)
            label = data[1].to(device)
        elif isinstance(data, dict):
            image = data['image'].type(torch.FloatTensor).to(device)
            label = data['label'].to(device)
        else:
            print(type(data))
            raise TypeError
        
        with torch.no_grad():
            output = model(image)
            pred = nn.Softmax(dim=1)(output)
        
        real_label.append(label.cpu().numpy())
        pred_label.append(pred.argmax(1).cpu().numpy())

        if ii == 0:
            bs = len(label.cpu().numpy().tolist())
            # print(f'label:{len(real_label)}')
            # print(f'label.numpy():{real_label}')
            # print(f'pred.argmax(1):{pred_label}')

        correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    real_label = np.array(real_label)
    pred_label = np.array(pred_label)

    stacked = torch.stack((torch.tensor(real_label.flatten(), dtype=torch.int64), torch.tensor(pred_label.flatten(), dtype=torch.int64)), dim=1)
    # print(stacked[0:7, :])
    names = list(par.data_type_dict.values())
    # print(names)
   
    cm = torch.zeros(len(par.data_type_dict), len(par.data_type_dict), dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cm[tl, pl] = cm[tl, pl] + 1
    # print(cm)
    plot_confusion_matrix(cm, names)
    pres = "cm_"
    plt.savefig(f"{path.trainednet_path}/{pres + model.__class__.__name__ + trained_name}.png")
    # plt.show()
    # print(f'stacked.shape:{stacked.shape}')
    # print(f'len(real_label):{len(real_label)}') 
    # print(f'len(real_label):{len(real_label)}') 
    print(f'accurency = {correct}/{len(data_set_test) * bs} = {correct/len(data_set_test)/bs}')




