

import os
import dataset
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


path_dict = dataset.get_dataset_path()
pic_list = []   ### a list of torchvision.transforms.methodxx
# dataset.create_h5_file(os.path.join(path_dict["path"], path_dict["dict"][0]), pic_trans = pic_list)
data = dataset.h5py_dataset(os.path.join("/home/jzm/workspace/final/data_h5", (path_dict["dict"][0] + '.hdf5')))
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