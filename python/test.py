import torch
from torchvision import transforms
import os
import dataset
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import platform

system_name = platform.system()
if system_name == "Windows":
    path = "D:/workspace/art/pic/protocol/"
elif system_name == "Linux":
    path = "/home/jzm/workspace/final/pic/protocol/"


path_dict = dataset.get_dataset_path(path=path, print_ctr=1)
# pic_list = [
#         # transforms.Resize((96 * 2,96 * 2)),
#         # transforms.CenterCrop((400,400)),
#         transforms.ConvertImageDtype(torch.double),
#         transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
#         # transforms.Grayscale()
#     ]  
### a list of torchvision.transforms.methodxx

pic_list = [];
dataset.create_h5_file(os.path.join(path_dict["path"], path_dict["dict"][0]), pic_trans = pic_list)

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