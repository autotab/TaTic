import random
import numpy as np
import pandas as pd
import Get_filename as GF


def main(file_dir,num_packs1,num_packs2=32):
    dict1 = {}
    classname = []
    dirs = GF.get_name(file_dir)
    def read_data(dirs,num_packs):
        text = []
        for name in dirs:
            temp = pd.read_csv(name, header=None)
            temp = temp.values
            temp_data = temp[:, 1:2]
            idex = -1
            for e in temp_data:
                idex = idex + 1
                e1 = e[0].strip('[]').replace("'", "").replace(" ", "").split(',')
                if len(e1) < 3 * num_packs:
                    e1 = e1 + [0] * (3 * num_packs - len(e1))
                text.append([temp[idex][0], np.array(np.array(e1)[0:3 * num_packs], 'float32'), temp[idex][-1]])
        return text


    for name in dirs:
        name1=name.split('/')[-2]
        if name1 not in dict1:
            classname.append(name1)
            dict1[name1]=[]
            dict1[name1].append(name)
        else:
            dict1[name1].append(name)

    def scramble_data(text):
        cc = list(zip(text))
        random.seed(100)
        random.shuffle(cc)
        text[:] = zip(*cc)
        return text[0]


    short_all_data=[]
    long_all_data=[]
    for name in classname:
        short_data = read_data(dict1[name],num_packs1)
        long_data = read_data(dict1[name], num_packs2)
        short_da = scramble_data(short_data)
        short_da=short_da[:5000]
        short_all_data+=short_da

        long_da = scramble_data(long_data)
        long_da=long_da[:5000]
        long_all_data+=long_da
    return short_all_data,long_all_data
