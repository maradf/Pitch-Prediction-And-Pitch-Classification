# Written by Mara Fennema
#
# Class definition of the LSTM model used

import torch
import torch.nn as nn
from torch.autograd import Variable


class myLSTM(nn.Module):
    def __init__(self, in_features, out_features, batch_size, num_layers):
        super(myLSTM, self).__init__()
        self.lstm = nn.LSTM(in_features, out_features, num_layers, batch_first=True)
        self.batchSize = batch_size

    def forward(self, inputs, returnHidden=False):
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.

        out, hidden = self.lstm(inputs.view(inputs.size(0), -1, 1))
        if returnHidden:
            return out, hidden
        else:
            return out

    def generate(self, inputs, hidden=None):
        if hidden:
            outputs, hidden = self.lstm(inputs.view(inputs.size(0), -1, 1), hidden)
        else:
            outputs, hidden = self.lstm(inputs.view(inputs.size(0), -1, 1))
        return outputs, hidden

    def getHidden(self):
        return self.hidden

    def loss(self, inputs, outputs):
        lossvector = abs(outputs - inputs.view(inputs.size(0), -1, 1))
        currentLoss = torch.mean(lossvector.mean(1).squeeze())
        return currentLoss
