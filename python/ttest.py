import torch
from torchvision import transforms
import os
import dataset
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import platform
import numpy as np
import random
from torch import optim
import model
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader


def train(model, data_set_training, optimizer, loss_fn, epoch):
    model.train()

    loss_total = 0

    for _, data in enumerate(data_set_training):
        if isinstance(data, list):
            image = data[0].type(torch.FloatTensor).to(device)
            label = data[1].to(device)
        elif isinstance(data, dict):
            image = data['image'].type(torch.FloatTensor).to(device)
            label = data['label'].to(device)
        else:
            print(type(data))
            raise TypeError
        
        optimizer.zero_grad()
        print(f'image.shape:{image.shape}')
        output = model(image)

        loss = loss_fn(output, label)
        loss_total+=loss.item()

        loss.backward()
        optimizer.step()
    
    print(f'{round(loss_total,2)} in epoch {epoch}')
    return loss_total


def test(model, data_set_test):
    model.eval()

    correct = 0

    for _, data in enumerate(data_set_test):

        if isinstance(data, list):
            image = data[0].type(torch.FloatTensor).to(device)
            label = data[1].to(device)
        elif isinstance(data, dict):
            image = data['image'].type(torch.FloatTensor).to(device)
            label = data['label'].to(device)
        else:
            print(type(data))
            raise TypeError
        
        with torch.no_grad():
            output = model(image)
            pred = nn.Softmax(dim=1)(output)
        
        correct+=(pred.argmax(x) == label).type(torch.float).sum().item()
    
    print(f'accurency = {correct}/{len(test_dataloader)*4} = {correct/len(test_dataloader)/4}')


device = "cuda" if torch.cuda.is_available() else "cpu"

if platform.system() == "Windows":
    h5file_path = "D:/workspace/art/data_h5"
else:
    h5file_path = "/home/jzm/workspace/final/data_h5"

def get_file_name(path):
    dict_name = []
    for cur_dir, dirs, files in os.walk(path):
        if (str(files) != '[]'):
            return {"path":path, "file":files}
        else:
            return "empty dictory"
    
pic_path_dict = get_file_name(h5file_path + '/')

file_type = []
data_type =[]
para_index = []

if isinstance(pic_path_dict, dict):
    print(f'path:{pic_path_dict["path"]}')
    print(f'file:{pic_path_dict["file"]}')
    file_list = pic_path_dict["file"]
    for file in file_list:
        file_type.append(file.split('_')[0])
        data_type.append(file.split('_')[1])
        para_index.append(file.split('_')[2])
else:
    print(pic_path_dict)

pic_trans = []
data_set_dict = {}
data_type_dict = {0:"tdma", 1:"aloha", 2:"csma", 3:"slottedaloha"}
# print(f'file_type:{file_type}')
# print(f'data_type:{data_type}')
# print(f'para_index:{para_index}')

for kk in range(0, 3, 2):

    if not os.path.exists(h5file_path + '/' + pic_path_dict["file"][kk]):
        dataset.create_h5_file(os.path.join(pic_path_dict["path"], pic_path_dict["file"][kk]), pic_trans = pic_list)

    data_set_dict[pic_path_dict["file"][kk]] = \
            dataset.h5py_dataset(os.path.join(h5file_path, (pic_path_dict["file"][kk])))
    # data = dataset.h5py_dataset(os.path.join(h5file_path, (pic_path_dict["dict"][kk] + ".hdf5")))

    dataset_scale = data_set_dict[pic_path_dict["file"][kk]].__len__()

    pic_index = int(np.random.randint(0, dataset_scale - 1, 1))

    imag, data_type = data_set_dict[pic_path_dict["file"][kk]].__getitem__(pic_index)
    data_type = data_type_dict[np.int16(data_type.numpy())]

    print(f'number of picture: {pic_index}')
    # print(f'data from: {pic_path_dict["dict"][kk]}')
    print(f'data type: {data_type}')
    # print(dataset_scale)
    # print(imag.shape)      ### for RGB pic: (3, x, y)
    # print(type(imag))      ### type:numpy.ndarray
    # ### show picture
    # imag = imag.transpose(1, 2, 0)   ### change (3, x, y) to (x, y, 3)
    # print(imag.shape)
    # plt.imshow(imag.astype('uint8'))
    # plt.show()  
    # print(f'{pic_path_dict["dict"][kk]} dataset_scale: {dataset_scale}')


for k in data_set_dict:
    if k.split('_')[1] == "training":
        data_set_training = data_set_dict[k]
    elif k.split('_')[1] == "test":
        data_set_test = data_set_dict[k]

data_set_training = DataLoader(data_set_training, batch_size=4, shuffle=True)
data_set_test = DataLoader(data_set_test, batch_size=4, shuffle=True)

model = model.conv_nn()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.5)
loss_fn = nn.CrossEntropyLoss()

EPOCH = 100

loss_all = []

for epoch in range(EPOCH):
    loss = train(model, data_set_training, optimizer, loss_fn, epoch=epoch)
    loss_all.append(loss)
    test(model, data_set_test)

    plt.plot(loss_all)
    plt.savefig(f"model_weights/{model.__class__.__name__}.png")
    plt.show()
    plt.close()

    torch.save(model.state_dict(), f"model_weights/{model.__class__.__name__}.pth")
    print("Saved PyTorch Model State to model.pth")


