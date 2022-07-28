from torch.utils.data import Dataset
import torch
import  numpy as np
class Mydataset(Dataset):
    def __init__(self, x, y,seq_leng):
        super(Mydataset, self).__init__()
        x=np.array(x[:,:seq_leng])
        self.x_data = x.reshape(-1, seq_leng, 1)
        self.x_data = torch.LongTensor(self.x_data)
        self.y_data = torch.LongTensor(y)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]