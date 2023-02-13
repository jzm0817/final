
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

        if "pic_enhance" in kwargs:
            self.data_enhance_list = kwargs["pic_enhance"]
            self.enhance_flag = True
        else:
            self.data_enhance_list = []
            self.enhance_flag = False
            
        self.transform = transforms.Compose(self.trans_ctr_list)
        self.data_enhance = transforms.Compose(self.data_enhance_list)
        self.set = []

        # print(self.root)
        type_dict = {"tdma":0, "aloha":1, "csma":2, "slottedaloha":3}

        for cur_dir, dirs, files in os.walk(self.root):
            for file in files:
                pic = read_image(os.path.join(cur_dir, file))
                pic = self.transform(pic)
                temp = file.split('.')[0]
                info = {
                    'image' : pic,
                    'label'  : torch.tensor(type_dict[temp.split('_')[0]])
                }
                self.set.append(info)

                if self.enhance_flag:
                    pic_ = self.data_enhance(pic)
                    info_ = {
                        'image' : pic_,
                        'label'  : torch.tensor(type_dict[temp.split('_')[0]])
                    }
                    self.set.append(info_)

                
    
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
            # print(pic_index)
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

    if "pic_enhance" in kwargs:
        data_enhance_list = kwargs["pic_enhance"]
        if len(data_enhance_list) > 0:
            enhance_flag = True
        else:
            enhance_flag = False
    else:
        # data_enhance_list = []
        enhance_flag = False


    if (len(kwargs) > 0) and ("nnpar" in kwargs):
        nnpar_index = kwargs["nnpar"]
    else:
        nnpar_index = 'x'
    # if "pic_size" in kwargs:
    #     pic_size = kwargs["pic_size"]
    # else:
    #     pic_size = [656, 875]
    
    h5file_suffix = '_nnpar' + str(nnpar_index)

    transform = transforms.Compose(trans_ctr_list)
    data_enhance = transforms.Compose(data_enhance_list)

    if platform.system() == 'Windows':
        path = "D:/workspace/art/data_h5"
    else:
        path_str = root.split('/')[:-3]
        path = '/home/jzm/workspace/final/data_h5'
        # for i in range(0, len(path_str)):
        #     path = os.path.join(path, path_str[i])
        # path = os.path.join(path, "data_h5")

    if not(os.path.exists(path)):
        os.makedirs(path)
    
    if enhance_flag:
        file_name = root.split('/')[-1] + h5file_suffix + '_a' + '.hdf5'
    else:
        file_name = root.split('/')[-1] + h5file_suffix + '.hdf5'

    save_path = path + '/' + file_name
    # print(f'h5file save_path:', save_path)
    data_set_data = []
    data_set_type = []
    if os.path.exists(save_path):
        os.remove(save_path)

    h5_file = h5py.File(save_path, "w")
    # print(root)
    for cur_dir, dirs, files in os.walk(root):
        # a = files[0].split('.')[0]
        for file in files:
            pic = read_image(os.path.join(cur_dir, file))
            pic = transform(pic)
            pic = np.array(pic).astype(np.float64)
            data_set_data.append(pic)
            if enhance_flag:
                pic_ = data_enhance(read_image(os.path.join(cur_dir, file)))
                pic_ = np.array(pic_).astype(np.float64)
                data_set_data.append(pic_)
            temp = file.split('.')[0]
            data_set_type.append(temp.split('_')[0])

    h5_file.create_dataset("image", data = data_set_data)
    h5_file.create_dataset("label", data = data_set_type)
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
        type_dict = {"tdma":0, "aloha":1, "csma":2, "slottedaloha":3}
        with h5py.File(self.file, 'r') as f:
            target = f['label'][index].decode()
            label = torch.tensor(type_dict[target])
            return f['image'][index], label

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
import torch
from torchvision import transforms
import os
import dataset
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import platform

path_dict = dataset.get_dataset_path()
pic_list = [
        # transforms.Resize((96 * 2,96 * 2)),
        # transforms.CenterCrop((400,400)),
        transforms.ConvertImageDtype(torch.double),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        # transforms.Grayscale()
    ]  
### a list of torchvision.transforms.methodxx
# dataset.create_h5_file(os.path.join(path_dict["path"], path_dict["dict"][0]), pic_trans = pic_list)

if platform.system() == 'Windows':
    path = 'D:/workspace/art/data_h5'
else:
    path = '/home/jzm/workspace/final/data_h5'


data = dataset.h5py_dataset(os.path.join(path, (path_dict["dict"][0] + '.hdf5')))
imag, data_type = data.__getitem__(0)
dataset_scale = data.__len__()  
print(dataset_scale)
print(imag.shape)      ### for RGB pic: (3, x, y)
print(type(imag))      ### type:numpy.ndarray
### show picture
imag = imag.transpose(1, 2, 0)   ### change (3, x, y) to (x, y, 3)
print(imag.shape)
plt.imshow(imag.astype('uint8'))
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
