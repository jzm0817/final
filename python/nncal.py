
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict

import path
import pickle
import net

index = 2

with open(path.nnpar_path + '/' + "par_" + str(index) + ".pkl", 'rb') as f:
    par = pickle.loads(f.read())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

model = par.ann
model = model.to(device)
summary(model, (3, 96 * 2, 96 * 2))


