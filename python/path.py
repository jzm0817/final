
import platform
import os

if platform.system() == "Windows":
    pic_path = "D:/workspace/art/pic/protocol/"
    h5file_path = "D:/workspace/art/data_h5"
    trainednet_path = "D:/workspace/art/net"

elif platform.system() == "Linux":
    pic_path = "/home/jzm/workspace/final/pic/protocol/"
    h5file_path = "/home/jzm/workspace/final/data_h5"
    trainednet_path = "/home/jzm/workspace/final/net"


'''
fun:
      find dictory with picture in default path or input path
      default path: origin_data_path_w for windows
                    origin_data_path_l for linux
para:
    print_ctr  -->
    path       -->
return:
    {
        path:xx      string
        dict:xxx     list
     }
'''

def get_dataset_path(path, **kwargs):

    if (len(kwargs) > 0) and ("print_ctr" in kwargs):
        print_flag = kwargs["print_ctr"]
    else:
        print_flag = 0
   
    origin_data_path = path

    if print_flag:
        print(f'deal with:', origin_data_path)

    ###  get pic dictory 
    dict_name = []
    for cur_dir, dirs, files in os.walk(origin_data_path):
        if (str(files) != '[]'):
            file_name = cur_dir.split('/')[-1]
            dict_name.append(file_name)

    if print_flag:
        print(f'pic dictory:', dict_name)

    return {"path":origin_data_path, "dict":dict_name}