

import os

pic_stft_path = "D:/workspace/art/pic/stft_origin/"

target_type = {}  ## awgn etc.
data_type = {}    ## training or test 


for cur_dir, dirs, files in os.walk(pic_stft_path):
    if (str(files) == '[]'):
        file_name = str(cur_dir.split('/')[-1])
    # print(file_name)
        pic_type = file_name.split('_')[0]
    # print(mod_type)
    # print("该目录下包含的文件：" + str(files))
    # print(str(files) == '[]')