
path:
windows:  D:/workspace/art
linux:    /home/jzm/workspace/final

以 windows 为例，路径中包含如下的文件夹
D:.
├─data_h5
├─data_info_mat
├─doc
├─matlab
│  └─rus
├─net
├─nninfo
├─pic
│  ├─protocol
│  │  ├─protocol_test_para1
│  │  ├─protocol_test_para2
│  │  ├─protocol_test_para4
│  │  └─protocol_training_para1
│  ├─protocol_training_para2
│  └─res
└─python
    ├─rus
    └─__pycache__

data_h5: 存放 .hdf5 格式的数据集文件，数据集根据 nninfo 中的部分图像调整参数及 pic 中的原始图片数据得到。
data_info_mat: 存放 matlab 生成的原始信号参数。
doc: 文档说明。
matlab: matlab 代码，其中 rus 文件夹存放废弃代码。
net: 存放训练好的网络，及测试结果的混淆矩阵截图。
nninfo: 存放网络或训练参数的 .pkl 文件。
pic: 存放 matlab 产生的原始图片。
python: python 代码，其中 rus 文件夹存放废弃代码。

Matlab 相关代码

complex_exponential_wave.m    复载波信号基本类
fh.m                          用于产生跳频信号类
generate_pic.m                保存时频图用的函数，适用于单一信道，或多信道单协议的情况
generate_pic_mul.m            保存时频图用的函数，适用于多信道多协议
get_files.m                   获取指定路径下的文件名的函数
get_pic.m                     生成时频图图片用的函数，调用即可在默认文加下生成 .jpg 格式图片
link16.m                      跳频频率集产生类
msk_modulation.m              MSK 调制类
para_est.m                    参数估计
pro_src_data.m                根据协议生成特定的信号  
psk_modulation.m              PSK 调制类
qam_modulation.m              QAM 调制类
rx_signal.m                   接收信号类
src_para.m                    产生基本信号参数
t.m                           测试用
test.m                        测试用
tfdec.m                       参数估计类
timeslot_est.m                时隙估计类 

产生不同协议的时频图方法:
简略:使用 src_para.m 保存 .mat 文件，在使用 get_pic 即可。具体细节如下：

## 参数设置
在 src_para.m 文件中设置基本的参数，文件中的主要参数如下:

save2mat        是否保存为 .mat 文件
index           若保存文件，index 为文件的编号，文件命名格式: {data_type}_para_{index}.mat
pic_number      设置需要产生的图片个数，<font color=red>在单一协议下，该参数表示一种协议产生的图片数，总图片数等于 4 * pic_number。在多协议的条件下，pic_number 为最终生成的图片数量。</font>
multi           多协议标志(0:单协议 1:多协议) 单协议或多协议指的是在一张时频图中出现的协议数量
freq_num        频点数量(信道数量)
rand_select     是否随机生成频率
data_type      "test"  or "training"

文件默认存放的路径为 data_info_mat 文件夹所在路径，已设置双系统下的默认路径

protocol_type   默认 4 种
package_len     协议中包的长度
mod_para        每个信道中用户的参数
fs              采样率
sample_length   样本总长度
slot_len        时隙长度
slot_info       时隙信息，对于 aloha 和 csma 无效
channel         信道信息，字符串，指定为高斯信道或衰落信道
snr             为高斯信道时指定信噪比

<font color=red>保存文件时，将 save2mat 置为 1，生成不同参数的文件时，记得修改 index 的数值</font>

### 单协议参数

单协议指每张时频图中只存在一种协议，频点个数可以是一个，也可以是多个。

#### 单一信道(频点)
<font color=red>multi  = 0、freq_num = 1</font>
rand_select 选择是否随机产生频率，为 0 时可设置频率值，(要修改在源代码 51 行处修改)，为 1 时从 link 16 的 51 个频点值中随机抽取。

#### 多信道(频点)
<font color=red>multi  = 1、 freq_num </font>
rand_select 选择是否随机产生频率，为 0 时可设置频率值，(要修改在源代码 51 行处修改)，为 1 时从 link 16 的 51 个频点值中随机抽取。

<center>
<table align="center"><tr>
<td><center><img src=aloha_2.jpg width="300" height="300"></center>
<center>单频点</center></td>
<td><center><img src=aloha_1.jpg width="300" height="300"></center>
<center>多频点</center></td>
</tr></table>
</center>

保存图片命名格式： {protocol_name}_{pic_number}.jpg
### 多协议参数

#### 固定频点
<font color=red>multi  = 1、freq_num、rand_select = 0</font>


#### 可变频点
<font color=red>multi  = 1、freq_num、rand_select = 1</font>

<center>
<table align="center"><tr>
<td><center><img src=aloha-slottedaloha-csma_4.jpg width="300" height="300"></center>
<center>固定频点</center></td>
<td><center><img src=aloha-slottedaloha-csma_2.jpg width="300" height="300"></center>
<center>可变频点</center></td>
</tr></table>
</center>

二者的区别在于：固定频点的所有图片使用的频点相同，可变频点则可能使用不同的频点。
注意命名格式：{protocol_name1}-{protocol_name2}_{pic_number}.jpg
每张图片包含多个协议，协议按照在图片中的位置由上到下依次命名，中间用 '-' 隔开。 
全过程的采样率涉及带通采样，选择不同的采样率会得到不同的等效频率，而选择的 610 MHz 采样率，得到的等效频率与原频率相比正好是相反的关系，即原频率越高，采样后的频率越低，具体细节可见相关文档，link16 类中包含等效频率的具体数值。

Python 相关代码

dataset.py  数据集制作及导入
ds.py       数据集加载及查看
gantest.py  源于网络
net.py      网络结构、训练、测试
nncal.py    测试网络结构及参数是否正确
nnpar.py    数据集制作中的图片处理参数设置、网络设置、训练参数测试
par.py      多模块共用参数存放处，命令行参数设置
path.py     多模块共用路径存放处
pic.py      图片处理函数，将图片有信号部分切割并移动到中心
plotcm.py   混淆矩阵绘制相关
t.py        测试
test.py     测试
tt.py       测试
ttest.py    主要内容