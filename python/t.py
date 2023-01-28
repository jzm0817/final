
import platform
import os
import matplotlib.pyplot as plt
from torchvision.io import read_image

system_name = platform.system()

if system_name == "Windows":
    print("windows")
elif system_name == "Linux":
    print("Linux")
else:
    print(system_name)

pic_stft_path = "D:/workspace/art/pic/stft_origin/"

def load_datasets():
    training_root = pic_stft_path
