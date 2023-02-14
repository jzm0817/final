import argparse
import par
import path

import ds
import os
import numpy as np
import platform
from torchvision.io import read_image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import h5py
import matplotlib.pyplot as plt


path_dict = path.get_dataset_path(path.pic_path)
print(path_dict["dict"])
root = path_dict["path"] + path_dict["dict"][2]

pic_list = [
            transforms.Resize((192, 192)),
            # transforms.CenterCrop((400,400)),                                                      
            # transforms.ConvertImageDtype(torch.double),
            # transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            # transforms.Grayscale()
        ]

transform = transforms.Compose(pic_list)
cnt = 1
label = []
row = 0
col = 0
interval = 20
exten = 20
label_dict = {}
for cur_dir, dirs, files in os.walk(root):
    # for file in files:
        # if cnt == 1:
    file = files[0]
    print(files[0])
    img = read_image(os.path.join(cur_dir, file))
    # img = transform(img)
    print(f'img.shape:{img.shape}')
    print(f'type(img):{type(img)}')
    img = img.numpy()
    temp = img.transpose(1, 2, 0)
    plt.imshow(temp.astype('uint8'))
    plt.figure()
    # print(f'img.shape:{img.shape}')
    # print(f'type(img):{type(img)}')
    # print(f'img.shape[0]:{img.shape[0]}')
    # print(f'img.shape[1]:{img.shape[1]}')
    # print(f'img.shape[2]:{img.shape[2]}')
    row = img.shape[1]
    col = img.shape[2]
    print(f'row:{row}')
    # print(f'img[0, 0, 1:10]:{img[0, 0, 1:10]}')
    for i in range(0, row):
        if np.sum(img[0, i, :]) < 254 * col:
            label.append(i)
        # print(np.sum(img[0, i, :]))
        # for j in range(0, col):
            # if img[0, i, j] < 254:
                # label.append(i)
    
    print(label)
    label_diff = np.diff(label)
    print(label_diff)
    index = []

    if len(index) == 0 and len(label) > 0:
        label_dict[0] = label
    elif len(index) > 0 and len(label) > 0:
        for i in range(0, len(label_diff)):
            if label_diff[i] > interval:
                index.append(i) 
        for i in range(0, len(index)):
            if i == 0:
                label_dict[i] = label[:index[i]]
            else:
                label_dict[i] = label[index[i-1] + 1:index[i]]
        label_dict[len(index)] = label[index[len(index) - 1] + 1:]
    
    for i in range(0, len(label_dict)):
        img_ = np.ones((3, row, col)) * 254
        loc = label_dict[i][-1] - label_dict[i][0] + 2 * exten
        print(loc)
        print(row // 2 - loc // 2)
        print(row // 2 + (loc - loc // 2))
        img_[:, row // 2 - loc // 2:row // 2 + (loc - loc // 2), :] = img[:,  label_dict[i][0] - exten: label_dict[i][-1] + exten, :]
        img_ = img_.transpose(1, 2, 0)   ### change (3, x, y) to (x, y, 3)
    
        plt.imshow(img_.astype('uint8'))
        plt.figure()
        imgt = torch.tensor(img_.transpose(2, 0, 1))
        print(f'imgt.shape:{imgt.shape}')
        imgt = transform(imgt)
        print(f'imgt.shape:{imgt.shape}')
        imgt_ = imgt.numpy()
        imgt_ = imgt_.transpose(1, 2, 0)
        plt.imshow(imgt_.astype('uint8'))

    plt.show()
    cnt+=1
# print(list(set(label)))
print(row)
print(col)
print(label)
print(label[0] - 20, label[-1] + 20)
print(f'root:{root}')