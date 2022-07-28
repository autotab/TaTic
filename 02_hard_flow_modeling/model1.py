import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        :param n_inputs: int
        :param n_outputs: int
        :param kernel_size: int
        :param stride: int
        :param dilation: int
        :param padding: int
        :param dropout: float
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        
        self.chomp1 = Chomp1d(padding)  
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu, self.dropout1,
                                 self.conv2, self.chomp2, self.relu, self.dropout2
                                 )
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
        

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        :param num_inputs: int
        :param num_channels: list, the channels of all TCN blocks
        :param kernel_size: int
        :param dropout: float
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  
            padding = dilation_size*(kernel_size-1)
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  
            out_channels = num_channels[i]  
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)   

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_channel, output_size, num_channels, kernel_size, dropout, vocab_text_size, seq_leng):
        super(TCN, self).__init__()
        self.seq_leng =seq_leng
        self.vocab_text_size=vocab_text_size
        self.con_embed = nn.Conv2d(1, input_channel, (1, self.vocab_text_size), 1)
        self.tcn = TemporalConvNet(input_channel, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.classify = nn.Sequential(
            nn.Linear(num_channels[-1]*2, num_channels[-1]),
            nn.Linear(num_channels[-1], output_size),
        )



    def forward(self, inputs):
        inputs= torch.transpose(inputs,2,1)
        new_inputs = F.one_hot(inputs, num_classes=self.vocab_text_size).float()
        new_inputs = self.con_embed(new_inputs).squeeze()
        '''
        TCN:
        Inputs have to have dimension (N, C_in, L_in)
        '''
        y1 = self.tcn(new_inputs)  
        out = self.classify(y1[:,:,-2:].reshape(y1.shape[0], -1))
        
        return out