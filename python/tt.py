

import os

pic_stft_path = "D:/workspace/art/pic/stft_origin/"

target_type = []  ## awgn etc.
# data_type = {}    ## training or test 


for cur_dir, dirs, files in os.walk(pic_stft_path):
    if (str(files) != '[]'):
        file_name = cur_dir.split('/')[-1]
        target_type.append(file_name)
print(target_type)
        # pic_type = file_name.split('_')[0]
        # print(pic_type)
        # data_type = file_name.split('_')[1]
        # print(data_type)
    # print("该目录下包含的文件：" + str(files))
    # print(str(files) == '[]')