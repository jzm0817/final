

import os

pic_stft_path = "D:/workspace/art/pic/stft_origin/"

for cur_dir, dirs, files in os.walk(pic_stft_path):
    print("====================")
    print("现在的目录：" + cur_dir)
    print("该目录下包含的子目录：" + str(dirs))
    print("该目录下包含的文件：" + str(files))
    print(str(files) == '[]')