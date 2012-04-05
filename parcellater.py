# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:38:00 2012

@author: sebastian

A Script for parcellation of functional data into networks based on functional
connectivity data
"""

import os
import sys
import numpy as np
import nibabel as nib
from sklearn.cluster import spectral_clustering as sc


def Loader(path, source=''):
    # takes in the path to the source and the source(filename) if specified
    # if no source is specified, this assumes that it is parsed directly
    # with the path
    img = nib.load(os.path.join(path, source))
    data = img.get_data().astype('float32')

    return ((img, data))


def Averager(functionalData, nodeMask, nodes):
    # this only works if the temporal dimension in the 4D functional file is
    # the 4th dimension

    nodeStack = np.array([], dtype='float32')
    for node in nodes:
        nodeVoxels = functionalData[nodeMask == node]
        nodeAverage = np.mean(nodeVoxels, axis=0)
        nodeStack = np.concatenate((nodeStack, nodeAverage[np.newaxis, ...]),
                                   axis=0)

    return nodeStack


def Similarity(nodeMat):
    # takes in a matrix of nodes times observations and calculates some sort
    # of similarity matrix
    # for now, I use correlation with a threshold

    similarityMat = np.corrcoef(nodeMat)
    similarityMat[similarityMat < 0.5] = 0

    return similarityMat


def Cluster(similarityMat, clusterSolutions):

    clust = sc(similarityMat, k=clusterSolutions)

    # as the cluster vector ranges from 0 to X, we add 1
    # otherwise there would be one cluster 0 which usually ist just
    # the nonbrain voxels

    identity_vector = clust + 1

    return identity_vector


def MaskGenerater(identVec, maskData):

    networkMask = np.zeros_like(maskData)

    run = 1
    for item in identVec:
        # for each node in the mask, assign the network identity value item
        # from the identity vector
        networkMask[maskData == run] = item

    fullMask = networkMask[np.newaxis, ...]

    for node in np.unique(identVec):
        localMask = np.zeros_like(maskData)
        # get an index of the nodes that are inside the current network
        # this corresponds to their position in the vector but here we assume
        # that their numeric position inside the vector corresponds to their
        # node ID in the mask too
        elementIndex = np.where(identVec == node)[0]

        for num in range(len(elementIndex)):
            temp = elementIndex[num]
            localMask[maskData == temp] = maskData[maskData == temp]

        fullMask = np.concatenate((fullMask, localMask[np.newaxis, ...]),
                                  axis=0)

    return (networkMask, fullMask)


def Processer(arguments):

    (maskData,
     funcAbsPath,
     funcName,
     subList,
     clustSol) = arguments

    nodes = np.unique(maskData)[1:]
    subStack = np.array([], dtype='float32')

    for sub in subList:
        (funcImg, funcData) = Loader(os.path.join(funcAbsPath, sub),
                                     source=funcName)

        nodeStack = Averager(funcData, maskData, nodes)
        subStack = np.concatenate((subStack, nodeStack[np.newaxis, ...]),
                                  axis=0)

    # average over all subjects
    avgStack = np.mean(subStack, axis=0)
    similarityMat = Similarity(avgStack)

    identVec = Cluster(similarityMat, clustSol)
    (networkMask, fullMask) = MaskGenerater(identVec, maskData)

    return (networkMask, fullMask)


def Main(batchFile, configFile, sysPath):
    # open batchfile containing the subject names
    batch = open(os.path.join(sysPath, batchFile))
    conf = open(os.path.join(sysPath, configFile)).readline().strip().split()
    (mPath,
     mSource,
     funcAbsPath,
     funcName,
     funcRelPath,
     outPath) = conf

    subList = []
    clustSol = 7
    maskImage, maskData = Loader(mPath, source=mSource)

    # read the batchfile line by line
    for line in batch:
        subConf = line.strip().split()
        subList.append(subConf[0])

    arguments = (maskData,
                 funcAbsPath,
                 funcName,
                 subList,
                 clustSol)

    (networkMask, fullMask) = Processer(arguments)

    header = maskImage.get_header()
    affine = maskImage.get_affine()
    # save the masks to files
    netNif = nib.Nifti1Image(networkMask, affine, header=header)
    fullNif = nib.Nifti1Image(fullMask, affine, header=header)
    nib.nifti1.save(netNif, (outPath
                             + '/'
                             + 'network_mask_'
                             + str(clustSol)
                             + 'nii.gz'))

    nib.nifti1.save(fullNif, (outPath
                              + '/'
                              + 'full_mask_'
                              + str(clustSol)
                              + 'nii.gz'))

    return (networkMask, fullMask)


# Boilerplate to call main():
if __name__ == '__main__':
    print 'Running in direct Call mode!'

    # Information from the configfile should be fetched here so Main() can
    # be cleaned up for modular use

    sysPath = os.path.abspath(os.curdir)
    batchFile = sys.argv[1]
    configFile = sys.argv[2]
    Main(batchFile, configFile, sysPath)