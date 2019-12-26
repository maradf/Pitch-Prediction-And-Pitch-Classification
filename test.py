# Written by Mara Fennema
#
# Test the LSTM defined in train.py


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import shutil


from models import myLSTM
from dataloader import dataLoader

# Setting constants
path = "PATH-TO-TESTSET"
batchSize = 128
nRuns = 1
pickUpPoint = 850
numLayers = 8
loadLocation = "PitchLSTM_iteration_4050.pt"
learningrate = 0.01

allFiles = os.listdir(path)
nFiles = len(allFiles)
nBatches = int(nFiles / batchSize) + nFiles % batchSize
batchRuns = nBatches * nRuns

# Initialize the dataloader
data = dataLoader(path, batchSize, nRuns, pickUpPoint, False)
batch, batchLengths = data.nextBatch()

# Load the LSTM
lstm = myLSTM(1, 1, batchSize, numLayers)
lstm.load_state_dict(torch.load(loadLocation))
lstm.eval()


losses = []
for i in range(pickUpPoint, batchRuns):
    batch = torch.FloatTensor(batch)
    batchLengths = torch.FloatTensor(batchLengths)
    out = lstm.forward(batchLengths)
    loss = lstm.loss(batch, out)
    floatLoss = float(loss)
    losses.append(floatLoss)
    # Save losses ever 25 iterations
    if i % 25 == 0:
        print("Calculating loss... {}/{}, {}%".format(i, batchRuns, round(i/batchRuns*100, 2)))
        ndLosses = np.asarray(losses)
        np.savetxt("testLosses.txt", ndLosses)
    batch, batchLengths = data.nextBatch()
