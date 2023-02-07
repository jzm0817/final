import path
import ds
import net
import ds
import par

import os
import platform
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np 


def main(args):

    if args.test:
        train_flag = False
    else:
        train_flag = True 

    if args.bs:
        batch_size = args.bs
    else:
        batch_size = args.default

    # if args.nn:
    #     nn = args.nn
    # else:
    #     nn = args.default

    print(f'train_flag:{train_flag}')
    print(f'batch_size:{batch_size}')
    # print(f'nn:{nn}')
    # print(f'type(nn):{type(nn)}')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_set_dict = ds.create_h5dataset(pic_size=par.pic_size, pic_list=par.pic_list)     ###
    # data_set_dict = path.get_dataset_path(path.h5file_path)                                 ###data_set_dictdata_set_dict

    # print(data_set_dict.keys())
    para_index = 0
    data_set_training, data_set_test = ds.load_dataset(data_set_dict, para_index)

    print(f'data_set_training length:{data_set_training.__len__()}')
    print(f'data_set_test length:{data_set_test.__len__()}')

    # batch_size = 16
    if platform.system() == "Windows":
        num_workers = 0
    elif platform.system() == "Linux":
        num_workers = 8
    
    data_set_training = DataLoader(data_set_training, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    data_set_test = DataLoader(data_set_test, batch_size=batch_size, shuffle=True)

    model = net.convnn()

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    loss_fn = nn.CrossEntropyLoss()

    EPOCH = 110


    loss_all = []

    trained_name = '_' + 'epoch' + str(EPOCH) + '_'+ 'para' + str(para_index) + par.h5file_suffix + '_' + "bs" + str(batch_size)

    if train_flag:
        for epoch in range(EPOCH):
            loss = net.train(model, data_set_training, optimizer, loss_fn, epoch, device)
            loss_all.append(loss)
            # test(model, data_set_test)
            if (epoch + 1) == EPOCH:
                temp = list(data_set_dict.keys())[1]
                temp = temp.split('.')[0]

                torch.save(model.state_dict(), f"{path.trainednet_path}/{model.__class__.__name__ + trained_name}.pth")
                print("Saved PyTorch Model State to model.pth")

                plt.plot(loss_all)
                plt.savefig(f"{path.trainednet_path}/{model.__class__.__name__ + trained_name}.png")
                # plt.show()
                # plt.close()

    model = net.convnn()            
    model.load_state_dict(torch.load(f"{path.trainednet_path}/{model.__class__.__name__ + trained_name}.pth"))
    model.to(device)
    net.test(model, data_set_test, device, batch_size, trained_name)





if __name__ == "__main__":
    args = par.default_argument_parser().parse_args()
    main(args)