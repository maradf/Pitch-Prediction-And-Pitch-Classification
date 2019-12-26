# Written by Mara Fennema
#
# Class to get batches of data from a larger folder. This allows for optimised
# runtime (by not extracting all the data into one variable), and is better for
# performance.
# All returned data-blocks are of the same length, by padding the shorter ones
# so that they're as long as the longest file.

import numpy as np
import os
import shutil


class dataLoader:
    def __init__(self, path, batchSize, nRuns, pickUpPoint, printFile):
        self.path = path
        self.files = os.listdir(path)
        self.batch = pickUpPoint
        self.batchSize = batchSize
        self.nRuns = nRuns
        self.currentRun = 0
        self.printFile = printFile

    def getCurrentBatchNumber(self):
        return self.batch

    def getPath(self):
        return self.path

    def getFiles(self):
        return self.files

    def getBatchSize(self):
        return self.batchSize

    def nextBatch(self):
        path = self.path
        currentNthBatch = self.batch
        nextNthBatch = currentNthBatch + 1
        batchSize = self.batchSize
        startElem = currentNthBatch * batchSize
        endElem = nextNthBatch * batchSize
        fileNames = self.files[startElem : endElem]

        # For the last batch, if there's not enough files left for a full batch
        # it shuffles the dataset and fills the rest of the batch with randomised
        # files from the dataset.
        while len(fileNames) < batchSize:
            np.random.shuffle(self.files)
            nMissingFiles = batchSize - len(fileNames)
            newFilenames = self.files[0:nMissingFiles]
            fileNames = fileNames + newFilenames
            self.batch = 0

        # Get longest file from the batch
        longestFileLength = -1
        longestFileName = ""
        fileLengths = []
        for f in fileNames:
            filePath = path + f
            content = open(filePath, "r")
            length = len(content.readlines())
            fileLengths.append(length)
            if length > longestFileLength:
                longestFileLength = length
                longestFileName = f

        fileLengthsMatrix = np.zeros((batchSize, longestFileLength))
        for i in range(0, batchSize):
            fileLengthsMatrix[i] = fileLengths[i]

        batch = []
        nthFile = 0
        for f in fileNames:
            if self.printFile:
                print(f)
            filePath = path + f
            paddedPitchData = []

            try:
                # Extract the pitch values from the datafile into an ndarray
                pitchData = np.loadtxt(filePath, delimiter="\t", skiprows=1, usecols=(2))
                # Puts 0's as padding before the pitchdata
                length = len(pitchData)
                nPaddings = longestFileLength - length
                paddedPitchData = np.pad(pitchData, (nPaddings, 0), 'constant', constant_values = (0))
            except:
                paddedPitchData = np.zeros(longestFileLength)
            # Adds file's pitch to the batch
            batch.append(paddedPitchData)
            nthFile += 1

        ndBatch = np.asarray(batch)
        self.batch += 1
        return ndBatch, fileLengthsMatrix
