import os
def get_name(src_dir):# 参数；原文件路径，保存路径，源文件格式，保存文件格式
    all_names1 = []
    file_dir = src_dir
    for files in os.walk(file_dir):
        #当前路径下所有非目录子文件
        for name in files[2]:
            temp_names = files[0] + '/' + name
            all_names1.append(temp_names)
    return all_names1