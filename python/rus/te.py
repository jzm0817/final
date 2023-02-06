import torch
import torch.nn as nn


# With square kernels and equal stride
# m = nn.ConvTranspose2d(16, 33, 3, stride=2)
ngf = 128
m = nn.Sequential(nn.ConvTranspose2d(in_channels=ngf, out_channels=ngf * 16, kernel_size=4, stride=1, padding=0, bias =False),
nn.BatchNorm2d(ngf * 16),
nn.ReLU(inplace=True), 
nn.ConvTranspose2d(in_channels=ngf * 16, out_channels=ngf * 8, kernel_size=4, stride=2, padding=1, bias =False),
nn.BatchNorm2d(ngf * 8),
nn.ReLU(inplace=True),
            # 输入一个8*8*ngf*4
nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1,bias=False),
nn.BatchNorm2d(ngf * 4),
nn.ReLU(inplace=True),

# 输入一个16*16*ngf*2
nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias =False),
nn.BatchNorm2d(ngf * 2),
nn.ReLU(inplace=True),
nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf * 1, kernel_size=4, stride=2, padding=1, bias =False),
nn.BatchNorm2d(ngf * 1),
nn.ReLU(inplace=True),
nn.ConvTranspose2d(in_channels=ngf * 1, out_channels=3, kernel_size=5, stride=3, padding=1, bias =False),

# Tanh收敛速度快于sigmoid,远慢于relu,输出范围为[-1,1]，输出均值为0
nn.Tanh(),)
input = torch.randn(1, ngf, 1, 1)
output = m(input)

print(output.shape)

ndf = 128
m1 = nn.Sequential(
           
            nn.Conv2d(in_channels=3, out_channels= ndf, kernel_size= 5, stride= 3, padding= 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace= True),

            # input:(ndf, 32, 32)
            nn.Conv2d(in_channels= ndf, out_channels= ndf * 2, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *2, 16, 16)
            nn.Conv2d(in_channels= ndf * 2, out_channels= ndf *4, kernel_size= 4, stride= 2, padding= 1,bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *4, 8, 8)
            nn.Conv2d(in_channels= ndf *4, out_channels= ndf *8, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(ndf *8),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *8, 4, 4)
            # output:(1, 1, 1)
            nn.Conv2d(in_channels= ndf *8, out_channels= 1, kernel_size= 4, stride= 1, padding= 0, bias=True),

            # 调用sigmoid函数解决分类问题
            # 因为判别模型要做的是二分类，故用sigmoid即可，因为sigmoid返回值区间为[0,1]，
            # 可作判别模型的打分标准
            nn.Sigmoid()
        )

print((m1(output)).shape)