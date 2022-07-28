import os
import random
import argparse

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import Data_Split as DS
from model1 import TCN
from Mydataset import Mydataset


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_channel', metavar='INPUT', type=int, default=64)
    parser.add_argument('-in_ker_num', metavar='INPUT', type=int, default=64)
    parser.add_argument('-layers', metavar='INPUT', type=int, default=4)
    parser.add_argument('-seq_len',  metavar='INPUT',type=int, default=16)
    parser.add_argument('-ker_size',  metavar='INPUT',type=int, default=13)
    parser.add_argument('-fold', metavar='INPUT', type=int, default=0)
    parser.add_argument('-CUDA', metavar='INPUT', type=str, default='3')
    parser.add_argument('-epochs', metavar='INPUT', type=int, default=100)
    parser.add_argument('-batch_size', metavar='INPUT', type=int, default=256)
    parser.add_argument('-model_name', metavar='INPUT', type=str, default='./in_channel_{}-in_kernel_{}-layers_{}-seq_len_{:0>2d}-ker_size_{:0>2d}-fold_{}')
    return parser.parse_args()


args = get_args()
max_acc=0 
output_size = 28  

dropout = 0.35  
save_dir = './save_models/'  
if not os.path.exists(save_dir): os.mkdir(save_dir)
vocab_text_size = 1500  


input_channel = args.input_channel  
num_channels = [args.in_ker_num] * args.layers  
num_packs = args.seq_len 
seq_leng = num_packs  
kernel_size = args.ker_size  
fold = args.fold 
CUDA_VISIBLE_DEVICES = args.CUDA  
num_epochs = args.epochs  
outname = args.model_name.format(input_channel, args.in_ker_num, args.layers, args.seq_len, kernel_size, fold) 
batch_size = args.batch_size

file_dir = "./new_data/"
data = DS.main(file_dir, num_packs)

def scramble_data(text):
    cc = list(zip(text))
    random.seed(100)
    random.shuffle(cc)
    text[:] = zip(*cc)
    return text[0]

da = scramble_data(data)
x=[]
for j in range(5):
    x.append(da[(len(da)*j)//5: (len(da)*(j+1))//5])
train=[]
valid=x[fold%5]
test=x[(fold+1)%5]
for i in range(2,5):
    train+=x[(i+fold)%5]

train_x=np.array(train,'int32')[:,1:]
train_y=np.array(train,'int32')[:,0]
valid_x=np.array(valid,'int32')[:,1:]
valid_y=np.array(valid,'int32')[:,0]
test_x=np.array(test,'int32')[:,1:]
test_y=np.array(test,'int32')[:,0]
new_text=[]
new_text.append(list(train_x))
new_text.append(list(train_y))
new_text.append(list(valid_x))
new_text.append(list(valid_y))
new_text.append(list(test_x))
new_text.append(list(test_y))


def data_load(new_data,seq_leng):
    train_dataloader = []
    valid_dataloader = []
    test_dataloader = []
    train_dataloader.append(
        DataLoader(Mydataset(np.array(new_data[0], dtype="int32"), np.array(new_data[1], dtype="int32"),seq_leng),
                    batch_size=batch_size,
                    shuffle=True, num_workers=0))
    valid_dataloader.append(
        DataLoader(Mydataset(np.array(new_data[2], dtype="int32"), np.array(new_data[3], dtype="int32"),seq_leng),
                    batch_size=batch_size,
                    shuffle=True, num_workers=0))
    test_dataloader.append(
        DataLoader(Mydataset(np.array(new_data[4], dtype="int32"), np.array(new_data[5], dtype="int32"),seq_leng),
                    batch_size=batch_size,
                    shuffle=True, num_workers=0))
    return [train_dataloader,valid_dataloader,test_dataloader]


def train(model, device, train_loader, optimizer, epochs, i, loss_fn):
    model.train()
    total_loss = 0
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)
    correct = 0.
    sum_num = 0.
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.squeeze()
        pre = model(data)
        y_predict = torch.cat([y_predict, torch.max(pre, 1)[1]], 0)
        y_true = torch.cat([y_true, target], 0)
        loss = loss_fn(pre.to(device), target.to(device)).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(target)
        
        sum_num += len(target)
        if idx % 100 == 99:
            print("Fold {} Train Epoch: {}, iteration: {}, Loss: {}".format(i, epochs, idx + 1, loss.item()))
    avg_loss = total_loss / sum_num
    
    y_true = y_true.cpu().numpy().tolist()
    y_predict = y_predict.cpu().numpy().tolist()

    y_true_trans = np.array(y_true)
    y_predict_trans = np.array(y_predict)

    acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
    train_acc = 100. * acc

    print("---------------------------------------------------")
    print("Fold {} epoch:{}  training Loss:{:.4f} training acc:{:.4f}".format(i, epoch, avg_loss, train_acc))


def valid(model, device, dev_loader, epoch, i, loss_fn , num_epochs, scheduler):
    model.eval()
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)
    total_loss = 0.
    correct = 0.
    sum_num = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(dev_loader):
            data, target = data.to(device), target.to(device)
            target = target.squeeze()
            
            
            pre = model(data)
            y_predict = torch.cat([y_predict, torch.max(pre, 1)[1]], 0)
            y_true = torch.cat([y_true, target], 0)
            loss = loss_fn(pre, target)
            total_loss += loss.item() * len(target)
            sum_num += len(target)
        avg_loss = total_loss / sum_num
        y_true = y_true.cpu().numpy().tolist()
        y_predict = y_predict.cpu().numpy().tolist()

        y_true_trans = np.array(y_true)
        y_predict_trans = np.array(y_predict)

        acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
        valid_acc = 100. * acc
    scheduler.step(avg_loss)
    print("Fold {} epoch:{} valid Loss:{:.4f} valid acc:{:.4f}".format(i, epoch, avg_loss, valid_acc))
    print("---------------------------------------------------")
    return acc


def test(device, test_loader, i,loss_fn,save_dir,outname):
    
    model = TCN(input_channel=input_channel, output_size=output_size, num_channels=num_channels, kernel_size=kernel_size,
                dropout=dropout,
                vocab_text_size=vocab_text_size,seq_leng=seq_leng).to(device)
    
    model_dir = save_dir + "{}.pth".format(outname)
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)
    total_loss = 0.
    correct = 0.
    sum_num = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pre = model(data)
            y_predict = torch.cat([y_predict, torch.max(pre, 1)[1]], 0)
            y_true = torch.cat([y_true, target], 0)
            target = target.squeeze()
            loss_fn = loss_fn.to(device)
            loss = loss_fn(pre, target)
            total_loss += loss.item() * len(target)
            sum_num += len(target)
        avg_loss = total_loss / sum_num
        y_true_list = y_true.cpu().numpy().tolist()
        y_predict_list = y_predict.cpu().numpy().tolist()

        y_true_trans = np.array(y_true_list)
        y_predict_trans = np.array(y_predict_list)
        acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
        test_acc = 100. * acc
    print("Fold {} test Loss:{:.4f} test acc:{:.4f}".format(i, avg_loss,test_acc))
    print("---------------------------------------------------")


if __name__ == '__main__':
    
    dataload = np.squeeze(data_load(new_text,seq_leng))
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
    
    model = TCN(input_channel=input_channel, output_size=output_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout,
                vocab_text_size=vocab_text_size, seq_leng=seq_leng).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=5e-3, weight_decay=0)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20, verbose=True, min_lr=1e-5)
    max_acc = 0
    for epoch in range(num_epochs):
        train(model, device, dataload[0], optimizer, epoch, int(fold)+1,criterion)
        valid_acc = valid(model, device, dataload[1], epoch, int(fold)+1,criterion, num_epochs, scheduler)
        if max_acc < valid_acc:
            max_acc = valid_acc
            torch.save(model.state_dict(), save_dir + "{}.pth".format(outname))
    test(device, dataload[2], int(fold)+1, criterion, save_dir, outname)
    