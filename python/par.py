
from torchvision import transforms
import torch
import argparse


data_type_dict = {0:"tdma", 1:"aloha", 2:"csma", 3:"slottedaloha"}
# h5file_suffix = '_' + str(pic_size[0]) + 'x' + str(pic_size[1])

trained_suffix = '_' + "trainpara"


def default_argument_parser():
    parser = argparse.ArgumentParser(description="pytorch-learning")
    parser.add_argument('--test', action="store_true", help="test model")
    parser.add_argument('--create', action="store_true", help="create h5 file")
    parser.add_argument('--train', action="store_true", help="train model")
    parser.add_argument('--mul', action="store_true", help="special test model")
    parser.add_argument('--show', action="store_true", help="show ds pic")
    parser.add_argument('--te', type=int, default=0)
    parser.add_argument('--tr', type=int, default=0)
    parser.add_argument('--npa', type=int, default=0)
    return parser