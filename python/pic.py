

import numpy as np
from torchvision.io import read_image
import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


def pic_move(pic_tensor):
    label = []
    row = 0
    col = 0
    interval = 20    ## multiple frequency 
    exten = 20       ## safe distance
    label_dict = {}

    pic_np = pic_tensor.numpy()
    row = pic_np.shape[1]
    col = pic_np.shape[2]

    for i in range(0, row):
        if np.sum(pic_np[0, i, :]) < (254 * col) * 0.95:
            label.append(i)

    label_diff = np.diff(label)
    index = []
    # print(f'label_diff:{label_diff}')
    for i in range(0, len(label_diff)):
        if label_diff[i] > interval:
            index.append(i) 

    if len(index) == 0 and len(label) > 0:
        label_dict[0] = label
    elif len(index) > 0 and len(label) > 0:
        for i in range(0, len(index)):
            if i == 0:
                label_dict[i] = label[:index[i]]
            else:
                label_dict[i] = label[index[i-1] + 1:index[i]]
        label_dict[len(index)] = label[index[len(index) - 1] + 1:]
    # print(f'len(label_dict):{len(label_dict)}')

    img_tensor_dict = {}
    for i in range(0, len(label_dict)):
        img_ = np.ones((3, row, col)) * 254
        loc = label_dict[i][-1] - label_dict[i][0] + 2 * exten
        img_[:, row // 2 - loc // 2:row // 2 + (loc - loc // 2), :] = \
            pic_np[:,  label_dict[i][0] - exten: label_dict[i][-1] + exten, :]

        img_tensor_dict[i] = torch.tensor(img_)
    # print('--------------------------')
    return img_tensor_dict