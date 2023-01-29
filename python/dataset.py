
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

'''
fun:
      find dictory with picture in default path or input path
      default path: origin_data_path_w for windows
                    origin_data_path_l for linux
para:
    print_ctr  -->
    path       -->
return:
    {
        path:xx      string
        dict:xxx     list
     }
'''

def get_dataset_path(**kwargs):

    if (len(kwargs) > 0) and ("print_ctr" in kwargs):
        print_flag = kwargs["print_ctr"]
    else:
        print_flag = 0

    if (len(kwargs) > 0) and ("path" in kwargs):
        origin_data_path = kwargs["path"]
    else:
        ###  pic path
        origin_data_path_w = "D:/workspace/art/pic/stft_origin/"
        origin_data_path_l = "/home/jzm/workspace/final/pic/stft_origin"

        if platform.system() == 'Windows':
            origin_data_path = origin_data_path_w
        else:
            origin_data_path = origin_data_path_l
   
    if print_flag:
        print(f'origin_data_path:', origin_data_path)

    ###  get pic dictory 
    dict_name = []
    for cur_dir, dirs, files in os.walk(origin_data_path):
        if (str(files) != '[]'):
            file_name = cur_dir.split('/')[-1]
            dict_name.append(file_name)

    if print_flag:
        print(f'pic dictory:', dict_name)

    return {"path":origin_data_path, "dict":dict_name}

'''
fun: generate cls_dataset according to root or input para (option) 
     called by function 'load_dataset(path, **kwargs)'
para:
    root       -->
    (option)
    print_ctr  -->   
    pic_trans  -->
return:
    class cls_dataset
'''

class cls_dataset(Dataset):

    def __init__(self, root, **kwargs) -> None:
        super().__init__()
    
        if (len(kwargs) > 0) and ("print_ctr" in kwargs):
            print_flag = kwargs["print_ctr"]
        else:
            print_flag = 0

        self.root = root

        if print_flag:
            print(f'path is:', self.root)
            print(f'dictory is:', self.root.split('/')[-1])

        if "pic_trans" in kwargs:
            self.trans_ctr_list = kwargs["pic_trans"]
        else:
            self.trans_ctr_list = []
        self.transform = transforms.Compose(self.trans_ctr_list)
        self.set = []

        for cur_dir, dirs, files in os.walk(self.root):
            for file in files:
                pic = read_image(os.path.join(cur_dir, file))
                pic = self.transform(pic)
                info = {
                    'image' : pic,
                    'type'  : self.root.split('/')[-1]
                }
                self.set.append(info)
    
    def __getitem__(self, index):
        return self.set[index]
    
    def __len__(self):
        return len(self.set)


'''
fun:
    generate cls_dataset
para:
    path       -->
    (option)
    show_info  -->
    show_pic   -->
    print_ctr  -->   (used by 'class cls_dataset(Dataset)')
    pic_trans  -->   (used by 'class cls_dataset(Dataset)')
return:
    class cls_dataset
'''

def load_dataset(path, **kwargs):

    if (len(kwargs) > 0) and ("show_pic" in kwargs):
        show_pic = kwargs["show_pic"]
        del kwargs["show_pic"]
    else:
        show_pic = 0

    if (len(kwargs) > 0) and ("show_info" in kwargs):
        show_info = 1
        kwargs.update({"print_ctr":1})
    else:
        show_info = 0

    data_set = cls_dataset(root=path, **kwargs)

    if show_info:
        print(f'length of data set:', len(data_set))
        print(f'type of data set:', data_set.__getitem__(0)['type'])

    if show_pic:
        if (len(kwargs) > 0) and ("index" in kwargs):
            pic_index = kwargs["index"]
            print(pic_index)
            ### show picture
            for i in range(0, len(pic_index)):
                p = data_set.__getitem__(pic_index[i])['image']
                # print(p.shape)
                img = ToPILImage()(p)
                img.show()

    return data_set

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

def create_h5_file(root, **kwargs):

    if "pic_trans" in kwargs:
        trans_ctr_list = kwargs["pic_trans"]
    else:
        trans_ctr_list = []
    transform = transforms.Compose(trans_ctr_list)

    if platform.system() == 'Windows':
        path = "D:/workspace/art/data_h5"
    else:
        path_str = root.split('/')[:-3]
        path = '/'
        for i in range(0, len(path_str)):
            path = os.path.join(path, path_str[i])
        path = os.path.join(path, "data_h5")

    if not(os.path.exists(path)):
        os.makedirs(path)
    file_name = root.split('/')[-1] + '.hdf5'
    save_path = path + '/' + file_name

    data_set_data = []
    data_set_type = []
    if os.path.exists(save_path):
        os.remove(save_path)

    h5_file = h5py.File(save_path, "w")

    for cur_dir, dirs, files in os.walk(root):
        for file in files:
            pic = read_image(os.path.join(cur_dir, file))
            pic = transform(pic)
            pic = np.array(pic).astype(np.float64)
            data_set_data.append(pic)
            data_set_type.append(root.split('/')[-1])

    h5_file.create_dataset("image", data = data_set_data)
    h5_file.create_dataset("type", data = data_set_type)
    h5_file.close()  
    print(f'finish complete:', save_path.split('/')[-1])  

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
        with h5py.File(self.file, 'r') as f:
            return f['image'][index], f['type'][index].decode()

    def __len__(self):
        with h5py.File(self.file, 'r') as f:
            return len(f['image'])
    
'''
exapmle code:
1. use function 'load_dataset(path, **kwargs)'   ('class cls_dataset(Dataset)')

path_dict = get_dataset_path()
data_set = load_dataset(os.path.join(path_dict["path"], path_dict["dict"][0]))
p = data_set.__getitem__(0)['image']
print(data_set.__len__())
print(p.shape)       ### for RGB pic: torch.Size([3, x, y])
print(type(p))       ### type:torch.Tensor
img = ToPILImage()(p)
img.show()

2. use 'class h5py_dataset(Dataset)'
path_dict = get_dataset_path()
pic_list = []   ### a list of torchvision.transforms.methodxx
# create_h5_file(os.path.join(path_dict["path"], path_dict["dict"][0]), pic_trans = pic_list)
data = h5py_dataset(os.path.join("D:/workspace/art/data_h5", (path_dict["dict"][0] + '.hdf5')))
imag, data_type = data.__getitem__(0)
dataset_scale = data.__len__()  
print(dataset_scale)
print(imag.shape)      ### for RGB pic: (3, x, y)
print(type(imag))      ### type:numpy.ndarray
### show picture
imag = imag.transpose(1, 2, 0)   ### change (3, x, y) to (x, y, 3)
print(imag.shape)
plt.imshow(imag)
plt.show()   


notice:
function 'load_dataset(path, **kwargs)' is uesd to generate dataset according to picture directly
(or use 'class cls_dataset(Dataset)' to get dataset)

the way to show picture is in function 'load_dataset(path, **kwargs)':
p = data_set.__getitem__(pic_index[i])['image']
print(p.shape)
img = ToPILImage()(p)
img.show()

'class h5py_dataset(Dataset)' is used to get dataset from .hdf5 file
the way to show picture is complex

'''
