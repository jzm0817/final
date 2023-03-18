import argparse
import par
import path


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
from pic import pic_move

path_dict = path.get_dataset_path(path.pic_path)
print(path_dict["dict"])
root = path_dict["path"] + path_dict["dict"][6]

pic_list = [
            transforms.Resize((192, 192)),
            # transforms.CenterCrop((400,400)),                                                      
            transforms.ConvertImageDtype(torch.double),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
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
    cnt = int(np.random.randint(0, len(files), 1))
    file = files[cnt]
    print(f'cnt:{cnt}')
    print(f'file name:{files[cnt]}')
    img = read_image(os.path.join(cur_dir, file))
    img_show = img.numpy().transpose(1, 2, 0)
    plt.imshow(img_show.astype('uint8'))
    plt.figure()
    img_dict = pic_move(img, 1)
    print(f'len(img_dict):{len(img_dict)}')
    for i in range(0, len(img_dict)):
        img = img_dict[i]
        # print(f'img_dict:{img_dict}')
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        plt.imshow(img.astype('uint8'))
        plt.figure()
        img_ = img_dict[i]
        img_ = transform(img_)
        img_ = img_.numpy().transpose(1, 2, 0)
        plt.imshow(img_.astype('uint8'))
        plt.figure()

plt.show()