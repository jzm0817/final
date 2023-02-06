
import os
from torchvision.io import read_image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import numpy as np

class cls_dataset(Dataset):
    
    def __init__(self, root, all_type) -> None:
        super().__init__()
        self.root = root
        self.all_type = all_type
        ctr_dict = [
                # transforms.Resize((96 * 2,96 * 2)),
                # transforms.CenterCrop((400,400)),
                transforms.ConvertImageDtype(torch.double),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                transforms.Grayscale()
            ]
        self.transform = transforms.Compose(ctr_dict)
        self.set = []

        for cur_dir, dirs, files in os.walk(self.root):
            target_type = "awgn"
            if target_type in self.all_type:
                for file in files:
                    pic = read_image(os.path.join(cur_dir, file))
                    pic = self.transform(pic)
                    information = {
                        'image':pic,
                        'type' :target_type
                    }
                    self.set.append(information)
        
    def __getitem__(self, index):
        return self.set[index]
    
    def __len__(self):
        return len(self.set)


pic_stft_path = "D:/workspace/art/pic/stft_origin/"
def load_datasets(flag):
    training_root = pic_stft_path + "awgn_training"
    test_root = pic_stft_path + "awgn_test"

    all_type = ["awgn"]

    training_set = cls_dataset(root = training_root, all_type = all_type)
    test_set = cls_dataset(root = test_root, all_type = all_type)
    # print(len(training_set))

    if flag:
        print(len(training_set))
        print(len(test_set))

        print(training_set.__getitem__(1)['type'])
        p = training_set.__getitem__()['image']
        img = ToPILImage()(p)
        img.show()

    return training_set
    # training_dataloader = DataLoader(training_set, batch_size = 4)
    # test_dataloader = DataLoader(test_set, batch_size = 4)


l = load_datasets(1)
# print(type(l))