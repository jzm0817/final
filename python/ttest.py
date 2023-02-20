import path
import ds
import net
import ds
import par

import os
import platform
import pickle
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np 


def main(args):

    total_type = "protocol"

    if args.show:
        show_flag = True
    else:
        show_flag = False

    para_index_tr = args.tr
    para_index_te = args.te
    nnpar_index = args.npa

    trained_name = '_'+ 'para' + str(para_index_tr) + '_' + "nnpar_" + str(nnpar_index)
    test_name = '_'+ 'para' + str(para_index_tr) + '--' + str(para_index_te) + '_' + "nnpar_" + str(nnpar_index)


    with open(path.nnpar_path + '/' + "par_" + str(nnpar_index) + ".pkl", 'rb') as f:
        nnpar = pickle.loads(f.read())

    if args.create:
        create_file = True
    else:
        create_file = False

    if args.test:
        test_flag = True
        data_type = "test"
        para_index = para_index_te
        pic_enhance = []
        print("------------  testing  ------------")
    else:
        test_flag = False

    if args.train:
        train_flag = True 
        data_type = "training"
        para_index = para_index_tr
        pic_enhance = nnpar.pic_enhance_list
        print("------------  training  ------------")
    else:
        train_flag = False
    
    if args.mul:
        mul = True
    else:
        mul = False

    h5file_name = total_type + '_' + data_type + '_para' + \
        str(para_index) + "_nnpar" + str(nnpar_index) + '.hdf5'
    
    if mul:
        h5file_name = total_type + '_' + data_type + '_para' + \
        str(para_index) + "_nnpar" + str(nnpar_index) + '_mul.hdf5'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_set = ds.load_dataset_h5(file=h5file_name, pic_trans=nnpar.pic_list, \
        pic_enhance=pic_enhance, mul=mul, create_file=create_file,\
        show_pic=show_flag, index=[0, 1, 2]
    )     
    print(f'data set name   : {h5file_name}')
    print(f'data set type   : {data_type}')
    print(f'data set length : {data_set.__len__()}')
    if platform.system() == "Windows":
        num_workers = 0
        pin_mem = False
    elif platform.system() == "Linux":
        num_workers = 8
        pin_mem = True

    data_set = DataLoader(data_set, batch_size=nnpar.bs, shuffle=True, pin_memory=pin_mem, num_workers=num_workers, drop_last=False)
    
    # model = net.neuralnetwork(nnpar.nn_list).to(device)
    model = nnpar.ann.to(device)

    optimizer = optim.SGD(model.parameters(), lr=nnpar.lr, momentum=0.5)
    loss_fn = nn.CrossEntropyLoss()

    EPOCH = nnpar.epoch

    loss_all = []
    
    if train_flag:
        for epoch in range(EPOCH):
            loss = net.train(model, data_set, optimizer, loss_fn, epoch, device)
            loss_all.append(loss)
            # test(model, data_set_test)
            if (epoch + 1) == EPOCH:
                # temp = list(data_set_dict.keys())[1]
                # temp = temp.split('.')[0]

                torch.save(model.state_dict(), f"{path.trainednet_path}/nn{trained_name}.pth")
                print("Saved PyTorch Model State to model.pth")

                plt.plot(loss_all)
                plt.savefig(f"{path.trainednet_path}/nn{trained_name}.svg")
                # plt.show()
                # plt.close()

    # model = net.neuralnetwork(nnpar.nn_list)    
    if test_flag: 
        model = nnpar.ann    
        model.load_state_dict(torch.load(f"{path.trainednet_path}/nn{trained_name}.pth"))
        model.to(device)
        net.test(model, data_set, device, nnpar.bs, test_name)


if __name__ == "__main__":
    args = par.default_argument_parser().parse_args()
    main(args)