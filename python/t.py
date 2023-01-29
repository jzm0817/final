
import platform
import os
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
import numpy as np

system_name = platform.system()

if system_name == "Windows":
    print("windows")
elif system_name == "Linux":
    print("Linux")
else:
    print(system_name)

pic_stft_path = "D:/workspace/art/pic/stft_origin/"

def load_datasets():
    training_root = pic_stft_path + "awgn_training"
    test_root = pic_stft_path + "awgn_test"
    training_set = {"awgn":[]}
    test_set = {"awgn":[]}
    pic_type = ["awgn"]

    for cur_dir, dirs, files in os.walk(training_root):
        target_type = "awgn"
        if target_type in pic_type:
            for file in files:
                pic = read_image(os.path.join(cur_dir, file))
                training_set["awgn"].append(pic)

    for target_type in pic_type:
        print(f'{target_type}:{len(training_set[target_type])}')
    

    for cur_dir, dirs, files in os.walk(test_root):
        target_type = "awgn"
        if target_type in pic_type:
            for file in files:
                pic = read_image(os.path.join(cur_dir, file))
                test_set["awgn"].append(pic)

    for target_type in pic_type:
        print(f'{target_type}:{len(test_set[target_type])}')
    print(type(test_set["awgn"][0]))
    # img = ToPILImage()(test_set["awgn"][0])
    # img.show()

load_datasets()
