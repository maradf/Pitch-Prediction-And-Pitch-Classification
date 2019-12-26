# Written by Mara Fennema
#
# Create a csv of all of the cell states for the LOHINEU dataset, with the
# classification added as well.

import os
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


from models import myLSTM
from dataloader import dataLoader

lastModel = "PitchLSTM_iteration_4050.pt"


pitchpath = "PATH-TO-LABELED-DATAFILES"
files = os.listdir(pitchpath)

# Set constants
numlayers = 8
batchsize = 1
nFiles = len(files)
nBatches = int(nFiles / batchsize) + nFiles % batchsize

# Initialise dataloader
data = dataLoader(pitchpath, batchsize, 1, 0, False)
batch, batchLengths = data.nextBatch()

# Initialise LSTM
lstm = myLSTM(1, 1, batchsize, numlayers)
lstm.load_state_dict(torch.load(lastModel))
lstm.eval()

#
allHiddens = {'hidden': [], 'class': []}
i = 0
for f in files:
    batch = torch.FloatTensor(batch)
    batchLengths = torch.FloatTensor(batchLengths)
    hidden = (torch.randn(numlayers, batchsize, 1),
                    torch.randn(numlayers, batchsize, 1))

    # Get class of file
    pitchClass = -1
    if "LO" in f:
        pitchClass = 0
    elif "NEU" in f:
        pitchClass = 1
    elif "HI" in f:
        pitchClass = 2
    else:
        pitchClass = 3

    # Get the hidden layer
    out, hidden = lstm.generate(batchLengths, hidden)
    hidden = hidden[1]
    hidden = hidden.tolist()
    allHiddens['hidden'].append(hidden)
    allHiddens['class'].append(pitchClass)

    # Print the progress
    if i % 20 == 0:
        print("At file {} out of {}, {}%".format(i, nFiles, round((i/nFiles)*100, 2)))
    i += 1
    batch, batchLengths = data.nextBatch()

# Save the dict to a csv
df = pd.DataFrame(allHiddens)
df.to_csv(path_or_buf="PATH-TO-CSV")
