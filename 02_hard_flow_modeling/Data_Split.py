import random
import numpy as np
import pandas as pd
import Get_filename_1 as GF
from sklearn import model_selection
def main(file_dir,num_packs):
    dict1 = {}
    classname = []
    dir = GF.get_name(file_dir)
    # 读取数据
    def read_data(dir):
        text = []
        for name in dir:
            temp = pd.read_csv(name, header=None)
            temp = temp.values
            temp_data = temp[:, 1:2]
            idex = -1
            for e in temp_data:
                idex = idex + 1
                e1 = e[0].strip('[]').replace("'", "").replace(" ", "").split(',')
                if len(e1) < 3 * num_packs:
                    e1 = e1 + [0, 0, 0] * ((3 * num_packs - len(e1)) // 3)
                else:
                    e1 = np.array(np.array(e1)[0:3 * num_packs], 'float32')
                temp1 = []
                temp1.append(temp[idex][-1])
                for i in range(num_packs):
                    temp1.append(int(e1[3 * i]))
                text.append(temp1)
        return text


    for name in dir:
        name1=name.split('/')[-2]
        if name1 not in dict1:
            classname.append(name1)
            dict1[name1]=[]
            dict1[name1].append(name)
        else:
            dict1[name1].append(name)

    # 数据打乱处理
    def scramble_data(text):
        cc = list(zip(text))
        random.seed(100)
        random.shuffle(cc)
        text[:] = zip(*cc)
        return text[0]


    all_data=[]
    for name in classname:
        data = read_data(dict1[name])
        da = scramble_data(data)
        if len(da)<=5000:
            da=da
        else:
            da=da[:5000]
        all_data+=da
    return all_data
