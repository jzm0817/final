
from torchvision import transforms
import torch

pic_size = [96 * 2, 96 * 2]
pic_list = [
            transforms.Resize((pic_size[0], pic_size[1])),
            # transforms.CenterCrop((400,400)),
            transforms.ConvertImageDtype(torch.double),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            # transforms.Grayscale()
        ]

data_type_dict = {0:"tdma", 1:"aloha", 2:"csma", 3:"slottedaloha"}
h5file_suffix = '_' + str(pic_size[0]) + 'x' + str(pic_size[1])

trained_suffix = '_' + "trainpara"