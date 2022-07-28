import argparse
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from model1 import TCN
from Mydataset import Mydataset
import Get_filename_1 as GF


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_channel', metavar='INPUT', type=int, default=64)
    parser.add_argument('-in_ker_num', metavar='INPUT', type=int, default=64)
    parser.add_argument('-layers', metavar='INPUT', type=int, default=4)
    parser.add_argument('-seq_len',  metavar='INPUT',type=int, default=32)
    parser.add_argument('-ker_size',  metavar='INPUT',type=int, default=13)
    parser.add_argument('-fold', metavar='INPUT', type=int, default=0)
    parser.add_argument('-CUDA', metavar='INPUT', type=str, default='3')
    parser.add_argument('-batch_size', metavar='INPUT', type=int, default=256)
    parser.add_argument('-model_name', metavar='INPUT', type=str, default='./all_model/in_channel_64-in_kernel_064-layers_4-seq_len_32-ker_size_13-fold_0.pth')
    parser.add_argument('-input_name', metavar='INPUT', type=str, default='decision-entropy-ntree_30-ratio_0.8-pack_3-fold_0')
    return parser.parse_args()


args = get_args()
output_size = 28  
max_words = 10000  
dropout = 0.35  
save_dir = './save_models/'  
vocab_text_size = 1500 
CUDA_VISIBLE_DEVICES=args.CUDA
input_channel = int(args.input_channel)  
num_channels = [int(args.in_ker_num)] * int(args.layers)  
num_packs =int(args.seq_len) 
seq_leng =num_packs  
kernel_size = int(args.ker_size)  
fold = args.fold 
outname = args.model_name 
input_name=args.input_name 


def read_data(file_dir,seq_leng):
    global valid_pre_labels,test_pre_labels,valid_true_labels,test_true_labels
    valid_data = []
    valid_labels = []
    test_data = []
    test_labels = []

    birth_data1=[]
    with open(file_dir + input_name +"_valid_pre_true.csv") as csvfile1:
        tempvalid = csv.reader(csvfile1)
        for row in tempvalid:  # load the data from get csv file into birth_data
            birth_data1.append(row)
    valid_pre_labels=np.array(birth_data1,"int32")[:,0]
    valid_true_labels = np.array(birth_data1,"int32")[:, 1]

    birth_data2=[]
    with open(file_dir + input_name +"_test_pre_true.csv") as csvfile2:
        temptest = csv.reader(csvfile2)
        for row in temptest:
            birth_data2.append(row)
            
    test_pre_labels=np.array(birth_data2,"int32")[:,0]
    test_true_labels = np.array(birth_data2,"int32")[:, 1]


    birth_data3=[]
    with open(file_dir + input_name +"_valid_test_data.csv") as csvfile3:
        tempdata = csv.reader(csvfile3)
        for row in tempdata:
            birth_data3.append(row)
    temp_valid_data=np.array(birth_data3)[0]
    for id1, e in enumerate(temp_valid_data):
        try:
            e1 = e.strip('[]').replace("'", "").replace(" ", "").split(',')
        except:
            break
        temp1 = []
        for i in range(seq_leng):
            if int(e1[3 * i])<0:
                temp1.append(0)
            else:
                temp1.append(int(e1[3 * i]))
        valid_data.append(temp1)
        valid_labels.append(int(birth_data3[1][id1]))
    temp_test_data=np.array(birth_data3)[2]
    for id2, e in enumerate(temp_test_data):
        try:
            e1=e.strip('[]').replace("'","").replace(" ","").split(',')
        except:
            break
        temp1 = []
        for i in range(seq_leng):
            if int(e1[3 * i])<0:
                temp1.append(0)
            else:
                temp1.append(int(e1[3 * i]))
        test_data.append(temp1)
        test_labels.append(int(birth_data3[3][id2]))
    return valid_data, valid_labels, test_data, test_labels


def data_load(valid_data,valid_labels,test_data,test_labels,seq_leng):
    valid_dataloader = []
    test_dataloader = []
    valid_dataloader.append(
        DataLoader(Mydataset(np.array(valid_data, dtype="int32"), np.array(valid_labels, dtype="int32"),seq_leng),
                    batch_size=64,
                    shuffle=True, num_workers=0))
    test_dataloader.append(
        DataLoader(Mydataset(np.array(test_data, dtype="int32"), np.array(test_labels, dtype="int32"),seq_leng),
                    batch_size=64,
                    shuffle=True, num_workers=0))
    return [valid_dataloader,test_dataloader]


def valid(model, device, valid_loader, i,loss_fn):
    global valid_pre_labels,valid_true_labels
    model.eval()
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)
    total_loss = 0.
    no_aprt=0
    correct = 0.
    sum_num = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            try:
                pre = model(data)
                y_predict = torch.cat([y_predict, torch.max(pre, 1)[1]], 0)
                y_true = torch.cat([y_true, target], 0)
                target = target.squeeze()
                loss_fn = loss_fn.to(device)
                loss = loss_fn(pre, target)
                total_loss += loss.item() * len(target)
                sum_num += len(target)
            except:
                no_aprt = no_aprt + target.shape[0]
                print("[alter] {} samples didn't go through testing.".format(no_aprt))
        avg_loss = total_loss / sum_num
        y_true_list = y_true.cpu().numpy().tolist()
        y_predict_list = y_predict.cpu().numpy().tolist()

        y_true_trans = np.array(y_true_list)
        y_predict_trans = np.array(y_predict_list)
        acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
        valid_acc = 100. * acc
        y_true_trans1 = np.array(y_true_list+list(valid_true_labels))
        y_predict_trans1 = np.array(y_predict_list+list(valid_pre_labels))
        acc1 = balanced_accuracy_score(y_true_trans1, y_predict_trans1)
        valid_acc1 = 100. * acc1
    print("Testing fold {} loss:{:.4f}, acc:{:.4f}".format(i, avg_loss, valid_acc))
    print("Testing fold {} current total loss:{:.4f} current total acc:{:.4f}".format(i, avg_loss,valid_acc1))
    print("---------------------------------------------------")


def test(model, device, test_loader, i, loss_fn):
    global test_pre_labels, test_true_labels
    model.eval()
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)
    total_loss = 0.
    no_aprt = 0
    correct = 0.
    sum_num = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            try:
                pre = model(data)
                y_predict = torch.cat([y_predict, torch.max(pre, 1)[1]], 0)
                y_true = torch.cat([y_true, target], 0)
                target = target.squeeze()
                loss_fn = loss_fn.to(device)
                loss = loss_fn(pre, target)
                total_loss += loss.item() * len(target)
                sum_num += len(target)
            except:
                no_aprt=no_aprt+target.shape[0]
                print("[alter] {} samples didn't go through testing.".format(no_aprt))

        avg_loss = total_loss / sum_num
        y_true_list = y_true.cpu().numpy().tolist()
        y_predict_list = y_predict.cpu().numpy().tolist()

        y_true_trans = np.array(y_true_list)
        y_predict_trans = np.array(y_predict_list)
        matrix = confusion_matrix(y_true_trans, y_predict_trans)
        acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
        test_acc = 100. * acc

        y_true_trans1 = np.array(y_true_list+list(test_true_labels))
        y_predict_trans1 = np.array(y_predict_list+list(test_pre_labels))
        matrix1 = confusion_matrix(np.array(y_true_trans1), np.array(y_predict_trans1))
        acc1 = balanced_accuracy_score(y_true_trans1, y_predict_trans1)
        test_acc1 = 100. * acc1
    print("Valid fold {} loss:{:.4f} acc:{:.4f}".format(i, avg_loss,test_acc))
    print("Valid fold {} current total loss:{:.4f} current total acc:{:.4f}".format(i, avg_loss, test_acc1))
    print("---------------------------------------------------")



if __name__ == '__main__':
    file_dir = "data/"
    valid_data, valid_labels, test_data, test_labels = read_data(file_dir, seq_leng)
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

    dataload = data_load(valid_data,valid_labels,test_data,test_labels,seq_leng)
    dataload = np.squeeze(dataload)

    model = TCN(input_channel=input_channel, output_size=output_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout,
                vocab_text_size=vocab_text_size, seq_leng=seq_leng).to(device)

    model.load_state_dict(torch.load(outname, map_location=torch.device('cpu'))) 

    criterion = torch.nn.CrossEntropyLoss()

    valid(model, device, dataload[0], fold, criterion)
    test(model, device, dataload[1], fold, criterion)