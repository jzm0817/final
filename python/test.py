import torch
from torchvision import transforms
import os
import dataset
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import platform
import numpy as np
import random
import path

system_name = platform.system()
if system_name == "Windows":
    pic_path = "D:/workspace/art/pic/protocol/"
elif system_name == "Linux":
    pic_path = "/home/jzm/workspace/final/pic/protocol/"


pic_path_dict = path.get_dataset_path(pic_path, print_ctr=1)

# pic_size = [96 * 2 * 2, 96 * 2 * 2]

pic_size = [656, 875]

pic_list = [
        transforms.Resize((pic_size[0], pic_size[1])),
        # transforms.CenterCrop((400,400)),
        # transforms.ConvertImageDtype(torch.double),
        # transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        # transforms.Grayscale()
    ]  
# pic_list = []

if len(pic_list) == 0:
    pic_size = [656, 875]

print(f'pic_size:{pic_size}')
print(f'pic_list:{pic_list}')
### a list of torchvision.transforms.methodxx


if platform.system() == "Windows":
    h5file_path = "D:/workspace/art/data_h5"
else:
    h5file_path = "/home/jzm/workspace/final/data_h5"


data_set_dict = {}
data_type_dict = {0:"tdma", 1:"aloha", 2:"csma", 3:"slottedaloha"}
h5file_suffix = '_' + str(pic_size[0]) + 'x' + str(pic_size[1])
print(f'pic_path_dict["dict"]:{pic_path_dict["dict"]}')
# len(pic_path_dict["dict"])

for kk in range(0, 1):
    print('-------------------------')
    print(h5file_path + '/' + pic_path_dict["dict"][kk] + h5file_suffix)
    if not os.path.exists(h5file_path + '/' + pic_path_dict["dict"][kk] + h5file_suffix + '.hdf5'):
    # if True:
        dataset.create_h5_file(pic_path_dict["path"] + '/' + pic_path_dict["dict"][kk], pic_trans = pic_list, pic_size=pic_size)
    
    data_set_dict[pic_path_dict["dict"][kk] + h5file_suffix] = \
            dataset.h5py_dataset(h5file_path + '/' + pic_path_dict["dict"][kk] + h5file_suffix + '.hdf5')
    # data = dataset.h5py_dataset(os.path.join(h5file_path, (pic_path_dict["dict"][kk] + ".hdf5")))
    print(f'data_set_dict.key:{data_set_dict.keys()}')
    # print(f'pic_path_dict["dict"][kk]:{pic_path_dict["dict"][kk] + '_' + str(pic_size[0]) + 'x' + str(pic_size[1])}')
    dataset_scale = data_set_dict[pic_path_dict["dict"][kk] + h5file_suffix].__len__()
    print(f'dataset_scale:{dataset_scale}')
    pic_index = int(np.random.randint(0, dataset_scale - 1, 1))

    imag, data_type = data_set_dict[pic_path_dict["dict"][kk] + h5file_suffix].__getitem__(pic_index)
    data_type = data_type_dict[np.int16(data_type.numpy())]

    print(f'number of picture: {pic_index}')
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
    # print(f'{pic_path_dict["dict"][kk]} dataset_scale: {dataset_scale}')
print(f'data_set_dict:{data_set_dict.keys()}')
    