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



'''
fun:
    generate .hdf5 file according to original picture
para:
    root    -->
    (option)
    reserve
return:
    class cls_dataset
'''

def create_h5file(file_name, **kwargs):

    if "pic_trans" in kwargs:
        trans_ctr_list = kwargs["pic_trans"]
    else:
        trans_ctr_list = []
    
    if "pic_enhance" in kwargs:
        data_enhance_list = kwargs["pic_enhance"]
        if len(data_enhance_list) > 0:
            enhance_flag = True
        else:
            enhance_flag = False
    else:
        enhance_flag = False

    transform = transforms.Compose(trans_ctr_list)
    data_enhance = transforms.Compose(data_enhance_list)    
    file_path = path.h5file_path + '/' + file_name

    pic_path = path.pic_path + "_".join((file_name.split('.')[0]).split('_')[:-1])
    print(f'pic_path:{pic_path}')
    h5_file = h5py.File(file_path, "w")
    # print(root)
    data_set_data = []
    data_set_type = []
    for cur_dir, dirs, files in os.walk(pic_path + '/'):
        # a = files[0].split('.')[0]
        for file in files:
            pic = read_image(os.path.join(cur_dir, file))
            pic_dict = pic_move(pic)
            pic = pic_dict[0]
            pic = transform(pic)
            pic = np.array(pic).astype(np.float64)
            data_set_data.append(pic)
            temp = file.split('.')[0]
            data_set_type.append(temp.split('_')[0])
            if enhance_flag:
                pic_temp = read_image(os.path.join(cur_dir, file))
                pic_dict_ = pic_move(pic_temp)
                pic_ = pic_dict_[0]
                pic_ = transform(pic_)
                pic_ = data_enhance(pic_)
                pic_ = np.array(pic_).astype(np.float64)
                data_set_data.append(pic_)
                temp = file.split('.')[0]
                data_set_type.append(temp.split('_')[0])
            

    h5_file.create_dataset("image", data = data_set_data)
    h5_file.create_dataset("label", data = data_set_type)
    h5_file.close()  
    print(f'finish complete:', file_name)  

'''
fun:
    generate class h5py_dataset
para:
    root    -->
    (option)
    reserve
return:
    class h5py_dataset
'''

class h5py_dataset(Dataset):

    def __init__(self, file, **kwargs):
        super().__init__()
        self.file = file

    def __getitem__(self, index):
        type_dict = {"tdma":0, "aloha":1, "csma":2, "slottedaloha":3}
        with h5py.File(self.file, 'r') as f:
            target = f['label'][index].decode()
            label = torch.tensor(type_dict[target])
            return f['image'][index], label

    def __len__(self):
        with h5py.File(self.file, 'r') as f:
            return len(f['image'])
    

