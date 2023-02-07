
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


class trainpar():
    def __init__(self, batch_size, learning_rate, epoch,  nn_list):
        self.bs = batch_size
        self.lr = learning_rate
        self.epoch = epoch
        self.nn_list = nn_list


# ch_in = 32
# reduction = 16

# se_list = [
#     ("affine1", nn.Linear(ch_in, ch_in // reduction, bias=False)),
#     ("relu1", nn.ReLU(inplace=True)),
#     ("affine2", nn.Linear(ch_in // reduction, ch_in, bias = False)),
#     ("sigmoid", nn.Sigmoid()) 
# ]


# class se_block(nn.Module):
#     def __init__(self, ch_in, reduction=16):
#         super(se_block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(se_list)

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


# se_list = OrderedDict(se_list)

# se = se_block(ch_in)

# nn_list = [ ('conv1', nn.Conv2d(3, 10, kernel_size=5)),
#             ('max_pool1', nn.MaxPool2d(kernel_size=2)),
#             ('relu1', nn.ReLU(inplace=True)),
#             ('conv2', nn.Conv2d(10, 32, kernel_size=5)),
#             ('max_pool2', nn.MaxPool2d(kernel_size=2)),
#             ('relu2', nn.ReLU(inplace=True)),
#             ('dropout1', nn.Dropout2d()),
#             ('se_block', se),
#             ('flatten1', nn.Flatten(start_dim=1)),
#             ('affine1', nn.Linear(32 * 45 * 45, 50)),
#             ('affine2', nn.Linear(50, 10)),
#             ('affine3', nn.Linear(10, 4)),
#             ]

# nn_list = OrderedDict(nn_list)

class neuralnetwork(nn.Module):

    def __init__(self, nn_list):
        super().__init__()
        self.nn_list = nn_list
        self.ann = nn.Sequential(self.nn_list)
    
    def forward(self, x):
        return self.ann(x)



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
    
    if (epoch + 1) % 10 == 0:
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
    plot_confusion_matrix(cm, names, normalize=True)
    pres = "cm_"
    output_str = "_normal"
    plt.savefig(f"{path.trainednet_path}/{pres}nn{trained_name + output_str}.png")
    plt.figure()
    plot_confusion_matrix(cm, names)
    plt.savefig(f"{path.trainednet_path}/{pres}nn{trained_name}.png")
    # plt.show()
    # print(f'stacked.shape:{stacked.shape}')
    # print(f'len(real_label):{len(real_label)}') 
    # print(f'len(real_label):{len(real_label)}') 
    print(f'accurency = {correct}/{len(data_set_test) * bs} = {correct/len(data_set_test)/bs}')




