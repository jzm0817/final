
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


def create_h5dataset(**kwargs):
    pic_path_dict = path.get_dataset_path(path.pic_path)
    print(pic_path_dict)

    flag = False
    if len(pic_path_dict["dict"]) == 0:
        pic_path_dict = {"path":path.h5file_path}
        for cur_dir, dirs, files in os.walk(path.h5file_path):
            pic_path_dict = {"dict":files}
    else:
        flag = True
    
    print(pic_path_dict)

    if (len(kwargs) > 0) and ("pic_size" in kwargs):
        pic_size = kwargs["pic_size"]
    else:
        pic_size = [656, 875]

    if (len(kwargs) > 0) and ("pic_list" in kwargs):
        pic_list = kwargs["pic_list"]
    else:
        pic_list = []  

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

    data_set_dict = {}


    if flag:
        for kk in range(0, len(pic_path_dict["dict"])):

            if not os.path.exists(path.h5file_path + '/' + pic_path_dict["dict"][kk] + par.h5file_suffix + '.hdf5'):
                dataset.create_h5_file(pic_path_dict["path"] + '/' + pic_path_dict["dict"][kk], pic_trans = pic_list, pic_size=pic_size)
            
            data_set_dict[pic_path_dict["dict"][kk] + par.h5file_suffix] = \
                    dataset.h5py_dataset(path.h5file_path + '/' + pic_path_dict["dict"][kk] + par.h5file_suffix + '.hdf5')

            if show_pic:
                dataset_scale = data_set_dict[pic_path_dict["dict"][kk] + par.h5file_suffix].__len__()
                
                if not index:
                    pic_index = np.random.randint(low=0, high=dataset_scale - 1, size=pic_num)
                else:
                    pic_index = index

                for i in pic_index:
                    imag, data_type = data_set_dict[pic_path_dict["dict"][kk] + par.h5file_suffix].__getitem__(i)
                    data_type = par.data_type_dict[np.int16(data_type.numpy())]

                    print(f'number of picture: {i}')
                    print(f'data from: {pic_path_dict["dict"][kk]}')
                    print(f'data type: {data_type}')
                    # print(dataset_scale)
                    # print(imag.shape)      ### for RGB pic: (3, x, y)
                    # print(type(imag))      ### type:numpy.ndarray
                    # ### show picture
                    imag = imag.transpose(1, 2, 0)   ### change (3, x, y) to (x, y, 3)
                    print(imag.shape)
                    plt.imshow(imag.astype('uint8'))
                    plt.show()
    else: 
        for kk in range(0, len(pic_path_dict["dict"])):
 
            data_set_dict[pic_path_dict["dict"][kk]] = \
                        dataset.h5py_dataset(path.h5file_path + '/' + pic_path_dict["dict"][kk])

            if show_pic:
                dataset_scale = data_set_dict[pic_path_dict["dict"][kk]].__len__()
                
                if not index:
                    pic_index = np.random.randint(low=0, high=dataset_scale - 1, size=pic_num)
                else:
                    pic_index = index

                for i in pic_index:
                    imag, data_type = data_set_dict[pic_path_dict["dict"][kk]].__getitem__(i)
                    data_type = par.data_type_dict[np.int16(data_type.numpy())]

                    print(f'number of picture: {i}')
                    print(f'data from: {pic_path_dict["dict"][kk]}')
                    print(f'data type: {data_type}')
                    # print(dataset_scale)
                    # print(imag.shape)      ### for RGB pic: (3, x, y)
                    # print(type(imag))      ### type:numpy.ndarray
                    # ### show picture
                    imag = imag.transpose(1, 2, 0)   ### change (3, x, y) to (x, y, 3)
                    print(imag.shape)
                    plt.imshow(imag.astype('uint8'))
                    plt.show()


    if isinstance(data_set_dict, dict):
        return data_set_dict
    else:
        raise ValueError ("ds -> fun:create_h5dataset -> reuturn is not a dict")


'''
example code:
'''
# create_h5dataset(pic_size=pic_size, pic_list=pic_list)


def load_dataset(data_set_dict, data_set_index, data_set_type="both"):
    data_set_training = []
    data_set_test = []

    cnt = 0
    for k in data_set_dict:
        
        if (k.split('_')[1] == "training") and (int(str(k.split('_')[2])[-1]) == data_set_index):
            data_set_training = data_set_dict[k]
        elif (k.split('_')[1] == "test") and (int(str(k.split('_')[2])[-1]) == data_set_index):
            data_set_test = data_set_dict[k]
        else:
            cnt += 1
        if cnt == len(list(data_set_dict.keys())):
            raise ValueError("invalid para_index!")

    if data_set_type == "both":
        return data_set_training, data_set_test
    elif data_set_type == "train" or data_set_type == "training":
        return data_set_training
    elif data_set_type == "test":
        return data_set_test
    else:
        raise ValueError("ds -> fun:load_dataset -> invalid para:data_set_type")
