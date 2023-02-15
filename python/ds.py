
import path
import dataset
import par

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToPILImage

import matplotlib.pyplot as plt
import numpy as np


def create_h5file(file, **kwargs):

    if not os.path.exists(path.h5file_path + '/' + file):
        dataset.create_h5file(file, **kwargs)


def load_dataset_h5(file, **kwargs):

    if (len(kwargs) > 0) and ("mul" in kwargs):
        mul = kwargs["mul"]
    else:
        mul = False

    if (len(kwargs) > 0) and ("create_file" in kwargs):
        create_file = kwargs["create_file"]
    else:
        create_file = False

    if create_file:
        if os.path.exists(path.h5file_path + '/' + file):
            os.remove(path.h5file_path + '/' + file)
            print("remove existed file!")

    if mul:
        if not os.path.exists(path.h5file_path + '/' + file):
            print("************ create special file now ************")
            dataset.create_h5file_mul(file, **kwargs)
    else:
        if not os.path.exists(path.h5file_path + '/' + file):
            print("************ create file now ************")
            dataset.create_h5file(file, **kwargs)

    print(f'************ loading {file} ************')
    data_set = dataset.h5py_dataset(path.h5file_path + '/' + file)
    print(data_set.__len__())

    if (len(kwargs) > 0) and ("show_pic" in kwargs):
        show_pic = kwargs["show_pic"]
    else:
        show_pic = False

    if (len(kwargs) > 0) and ("index" in kwargs):
        index = kwargs["index"]
    else:
        index = False
    
    if (len(kwargs) > 0) and ("pic_num" in kwargs):
        pic_num = kwargs["pic_num"]
    else:
        pic_num = 1

    if show_pic:
        dataset_scale = data_set.__len__()
        
        if not index:
            pic_index = np.random.randint(low=0, high=dataset_scale - 1, size=pic_num)
        else:
            pic_index = index

        for i in pic_index:
            imag, data_type = data_set.__getitem__(i)
            data_type = par.data_type_dict[np.int16(data_type.numpy())]

            print('-------------------------------------')
            print(f'number of picture: {i}')
            print(f'data from: {file}')
            print(f'data type: {data_type}')

            imag = imag.transpose(1, 2, 0)   ### change (3, x, y) to (x, y, 3)
            # print(imag.shape)
            plt.imshow(imag.astype('uint8'))
            plt.show()


    return data_set