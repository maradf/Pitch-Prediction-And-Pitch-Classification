# Written by Mara Fennema
#
# LSTM to train pitch prediction. Every fiftieth model is saved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam

import numpy as np
import os
import shutil

from models import myLSTM
from dataloader import dataLoader

# Setting constants
path = "PATH-TO-TRAININGSET"
batchSize = 128
nRuns = 100
pickUpPoint = 4550 # Set to the last iteration done, including skipped batches
numLayers = 8
saveLocation = "PATH-WHERE-TO-SAVE-THE-TRAINED-MODELS"
learningrate = 0.01

allFiles = os.listdir(path)
nFiles = len(allFiles)
nBatches = int(nFiles / batchSize) + nFiles % batchSize
batchRuns = nBatches * nRuns

# Initialise the dataloader
data = dataLoader(path, batchSize, nRuns, pickUpPoint, False)

# Check if the defined save location exists. If not, create it.
if not os.path.isdir(saveLocation):
    os.makedirs(saveLocation)

# Retreive the first batch of data
batch, batchLengths = data.nextBatch()

# Initialize the LSTM and its hidden layers
lstm = myLSTM(1, 1, batchSize, numLayers)
optimizer = Adam(lstm.parameters(), lr=learningrate)

# Load the most up to date model
modelPath ="PitchLSTM_iteration_4050.pt"
lstm.load_state_dict(torch.load(modelPath))
lstm.eval()


# Create the file with all the loss values
lossPath = "PATH.txt" # Set to path where the txt needs to be saved
fiftyLosses = []

# Train the model
for i in range(pickUpPoint, batchRuns):
    lstm.train()
    batch = torch.FloatTensor(batch)
    batchLengths = torch.FloatTensor(batchLengths)
    out = lstm.generate(batchLengths)
    loss = lstm.loss(batch, out)
    floatLoss = round(float(loss), 2)
    fiftyLosses.append(floatLoss)

    # Save every fiftieth model, and calculate average loss of
    # these fifty iterations
    if i % 50 == 0:
        averageLoss = np.average(fiftyLosses)
        print("Iteration: {}% {}/{}".format(round((i/batchRuns)*100, 2), i+1, batchRuns))
        print("Loss: {}".format(averageLoss))
        fiftyLosses = []
        savePath = saveLocation + "PitchLSTM_iteration_{}.pt".format(i)
        torch.save(lstm.state_dict(), savePath)

        # Save loss value to separate file
        lossf = open(lossPath, "a+")
        lossf.write("{}\n".format(averageLoss))
        lossf.close()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    batch, batchLengths = data.nextBatch()
