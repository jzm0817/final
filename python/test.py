import torch
from torchvision import transforms
import os
import dataset
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import platform

system_name = platform.system()
if system_name == "Windows":
    pic_path = "D:/workspace/art/pic/protocol/"
elif system_name == "Linux":
    pic_path = "/home/jzm/workspace/final/pic/protocol/"


pic_path_dict = dataset.get_dataset_path(pic_path, print_ctr=1)
# pic_list = [
#         # transforms.Resize((96 * 2,96 * 2)),
#         # transforms.CenterCrop((400,400)),
#         transforms.ConvertImageDtype(torch.double),
#         transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
#         # transforms.Grayscale()
#     ]  
### a list of torchvision.transforms.methodxx


if platform.system() == "Windows":
    h5file_path = "D:/workspace/art/data_h5"
else:
    h5file_path = "/home/jzm/workspace/final/data_h5"

pic_list = []
data_set_dick = {}

for kk in range(0, len(pic_path_dict)):
    if not os.path.exists(h5file_path + '/' + pic_path_dict["dict"][kk] + ".hdf5"):
    # if True:
        dataset.create_h5_file(os.path.join(pic_path_dict["path"], pic_path_dict["dict"][kk]), pic_trans = pic_list)

    data_set_dick[pic_path_dict["dict"][kk]] = \
            dataset.h5py_dataset(os.path.join(h5file_path, (pic_path_dict["dict"][kk] + ".hdf5")))
    # data = dataset.h5py_dataset(os.path.join(h5file_path, (pic_path_dict["dict"][kk] + ".hdf5")))
    imag, data_type = data_set_dick[pic_path_dict["dict"][kk]].__getitem__(40)
    dataset_scale = data_set_dick[pic_path_dict["dict"][kk]].__len__()  
    # print(dataset_scale)
    # print(imag.shape)      ### for RGB pic: (3, x, y)
    # print(type(imag))      ### type:numpy.ndarray
    # ### show picture
    # imag = imag.transpose(1, 2, 0)   ### change (3, x, y) to (x, y, 3)
    # # print(imag.shape)
    # plt.imshow(imag.astype('uint8'))
    # plt.show()  
    print(f'{pic_path_dict["dict"][kk]} dataset_scale: {dataset_scale}')
    print(f'{pic_path_dict["dict"][kk]} data_type: {data_type}')