from logarithmicSearch import logarithmicSearch
from createMacroblocks import createMacroblocks
import numpy as np


def motionCompensation(referenceFrame, targetFrame):
    macroblocksize = 64
    predictedBlocks = []
    motionVectors = []

    targetMacroblocks = createMacroblocks(targetFrame)
    for i in range(targetMacroblocks.shape[0]):
        for j in range(targetMacroblocks.shape[1]):
            motionVectorBegin = (i * macroblocksize, j * macroblocksize)
            indexBlock = (i * macroblocksize, j * macroblocksize)
            motionVectorEND, prediction = logarithmicSearch(referenceFrame, targetMacroblocks[i, j, :, :], indexBlock)
            predictedBlocks.append(prediction)
            motionVectors.append(motionVectorBegin + motionVectorEND)


    predictedBlocks = np.array(predictedBlocks).reshape(targetMacroblocks.shape)
    motionVectors = np.array(motionVectors, dtype=(int, 4)).reshape((targetMacroblocks.shape[0], targetMacroblocks.shape[1], 4))

    return predictedBlocks, motionVectors