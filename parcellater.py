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
    # for every node in the mask
    for node in nodes:
        # generates a vector of voxels in the current node
        nodeVoxels = functionalData[nodeMask == node]

        # average value for all timepoints in the node
        nodeAverage = np.mean(nodeVoxels, axis=0)
        if nodeStack.size == 0:
            nodeStack = nodeAverage[np.newaxis, ...]
        else:
            nodeStack = np.concatenate((nodeStack,
                                        nodeAverage[np.newaxis, ...]),
                                       axis=0)

    print 'nodeStack:', nodeStack.shape

    return nodeStack


def Similarity(nodeMat):
    # takes in a matrix of nodes times observations and calculates some sort
    # of similarity matrix
    # for now, I use correlation with a threshold

    similarityMat = np.corrcoef(nodeMat)
    # set correlation values below threshold to 0
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
        run += 1

    # mask has to be concatenated along the 4th dimension in order not to
    # confuse poor old fslview
    fullMask = networkMask[..., np.newaxis]
    # add the original mask to the file
    fullMask = np.concatenate((fullMask, maskData[..., np.newaxis]), axis=3)

    # node takes the values of the unique cluster identifiers in the
    # identity Vector
    for node in np.unique(identVec):
        # set local mask to zero for every cluster iteration
        localMask = np.zeros_like(maskData)
        # get an index of the nodes that are inside the current network
        # this corresponds to their position in the vector but here we assume
        # that their numeric position inside the vector corresponds to their
        # node ID in the mask too
        elementIndex = np.where(identVec == node)[0]
        print 'element Index', elementIndex

        for num in range(len(elementIndex)):
            temp = elementIndex[num]
            localMask[maskData == temp] = temp

        fullMask = np.concatenate((fullMask, localMask[..., np.newaxis]),
                                  axis=3)

    return (networkMask, fullMask)


def Processer(arguments):

    (maskData,
     funcAbsPath,
     funcName,
     subList,
     clustSol) = arguments

    # nodes are the unique values in the mask (without 0)
    nodes = np.unique(maskData)[1:]
    print 'nodes:', nodes
    subStack = np.array([], dtype='float32')

    for sub in subList:
        # print 'Running', sub
        # load the functional file for the current subject
        (funcImg, funcData) = Loader(os.path.join(funcAbsPath, sub),
                                     source=funcName)
        print 'funcImg.shape', funcImg.shape

        # returns a matrix of nodes by timepoints averaged node voxels
        nodeStack = Averager(funcData, maskData, nodes)
        # then stack this into a matrix of subjects by nodes by timepoints
        # for first loop
        if subStack.size == 0:
            subStack = nodeStack[np.newaxis, ...]
        # for every other loop
        else:
            subStack = np.concatenate((subStack, nodeStack[np.newaxis, ...]),
                                      axis=0)

    # average over all subjects
    # returns again a matrix of nodes by timepoints but averaged over all
    # subjects
    avgStack = np.mean(subStack, axis=0)
    # returns a similarity matrix of nodes by nodes similarity values
    # here we are using correlation
    similarityMat = Similarity(avgStack)
    print 'similarity mat', similarityMat.shape

    # returns a vector of length=number nodes that contains the cluster number
    # for every element
    identVec = Cluster(similarityMat, clustSol)
    print 'length identVector:', len(identVec)

    # returns a 3D networkmask and a 4D fullmask
    (networkMask, fullMask) = MaskGenerater(identVec, maskData)
    print ('maskshapes. networkmask:', networkMask.shape,
           'fullmask:', fullMask.shape)

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

    maskImage, maskData = Loader(mPath, source=mSource)

    subList = []
    clustSol = 7
    numberNodes = maskData.max()

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
                             + str(numberNodes)
                             + '_'
                             + str(clustSol)
                             + '_cluster'
                             + '.nii.gz'))

    nib.nifti1.save(fullNif, (outPath
                              + '/'
                              + 'full_mask_'
                              + str(numberNodes)
                              + '_'
                              + str(clustSol)
                              + '_cluster'
                              + '.nii.gz'))

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