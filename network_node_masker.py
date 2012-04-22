# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 17:18:43 2012

@author: sebastian

Script to intersect Two different masks
Mostly for node and network mask intersections
"""
import sys
import numpy as np
import nibabel as nib
from scipy import ndimage as nd


def MonotonChecker(maskData, maskName='mask'):
    # check if the values in the mask are continuous
    maskValues = np.unique(maskData[maskData != 0])
    maskSize = len(maskValues)

    fullRange = np.arange(1, maskValues.max() + 1, 1)
    rangeCheck = np.in1d(maskValues, fullRange)

    if np.where(rangeCheck == False)[0].size > 0:
        missing = (rangeCheck == False)
        print '\n##### ATTENTION #####'
        print 'Your', maskName, 'is not continuous!'
        print 'This means that the values in your mask are not continuous from'
        print '0 to the maximum value'
        print 'these values are missing:', maskValues[missing]
        print '\nIf you expected this, all is fine.'
        print 'If not, you should consider checking your mask'
        print '##### ATTENTION #####\n'
    else:
        print '\nYour', maskName, 'is continuous, carry on! '

    return (maskValues, maskSize)


def Masker(netwData, nodeData):
    # Doc

    netwValues, netwSize = MonotonChecker(netwData, maskName='network mask')
    nodeValues, nodeSize = MonotonChecker(nodeData, maskName='node mask')

    Storage = {}
    # create empty storage for the whole network (without node IDs)
    wholeNet = np.zeros_like(nodeData)

    for node in nodeValues:
        # create empty node storage
        tempNode = np.zeros_like(nodeData)
        storeNode = np.zeros_like(nodeData)

        # set values in the node to 1
        tempNode[nodeData == node] = 1
        storeNode[nodeData == node] = node

        # calculate center of mass
        centerOfMass = nd.measurements.center_of_mass(tempNode)
        x, y, z = centerOfMass
        networkId = int(netwData[x, y, z])

        if networkId == 0:
            print ('\nnode '
                   + str(node)
                   + ' has a center of mass outside the network')
            # Center of mass lies outside of Yeos mask. Now if the node
            # overlaps at least a little bit with any network, we could still
            # include it in the mask

            # so let's check how many of the node's voxels actually are
            # inside the network and how many are not

            # size of the node
            nodeSize = len(nodeData[nodeData == node])
            # network ID in node Voxels
            nodeVoxels = np.zeros_like(nodeData)
            nodeVoxels[nodeData == node] = netwData[nodeData == node]
            # number of node voxels in network
            nodeNetVoxels = len(nodeVoxels[nodeVoxels != 0])
            # node overlap with mask voxels
            nodeNetOverlap = nodeNetVoxels / float(nodeSize)

            nodeNetworks = np.unique(nodeVoxels)
            print ('these are networks, node '
                   + str(node)
                   + ' is a member of:'
                   + str(nodeNetworks))

            if nodeNetworks.max() == 0:
                # all voxels of this node are outside of the mask, so we can
                # safely get rid of it
                print 'really everything is in the void.'

            elif nodeNetOverlap < 0.3:
                # there are nodes in the mask but just not enough
                print 'not everything in the void, but less than 0.3'

            else:
                netList = np.array([])
                maxList = np.array([])
                print 'Mask overlap of node', str(node), 'is', nodeNetOverlap
                for network in nodeNetworks[nodeNetworks != 0]:
                    # number of voxels in this network
                    inNetworkVoxels = len(nodeVoxels[nodeVoxels == network])
                    percentage = float(inNetworkVoxels) / nodeSize
                    netList = np.append(netList, network)
                    maxList = np.append(maxList, inNetworkVoxels)
                    print ('percentage of overlap between node '
                           + str(node)
                           + ' and network '
                           + str(network)
                           + ' is '
                           + str(percentage))

                networkId = int(netList[maxList.argmax()])
                print 'biggest overlap is with', networkId

        if not str(networkId) in Storage:
            # Stroage element is still empty so first define it
            Storage[str(networkId)] = np.zeros_like(nodeData)

        tempNet = Storage[str(networkId)]
        tempNetNode = tempNode
        tempNetNode[tempNode != 0] = networkId

        # reassign the values
        Storage[str(networkId)] = tempNet + storeNode
        wholeNet = wholeNet + tempNetNode

    return (Storage, wholeNet)


def Main(networkFile, nodeFile):

    # get the data
    netwData = networkFile.get_data()
    nodeData = nodeFile.get_data()

    if netwData.shape == nodeData.shape:
        # do something
        print 'some stuff, all is good'
        Storage, wholeNet = Masker(netwData, nodeData)
        networks = Storage.keys()
        print networks, type(networks), np.sort(networks)
        print 'voxels in nowhere:', len(np.unique(Storage['0']))
        print 'voxels in nowhere:', len(np.argwhere(Storage['0'] != 0))

        # add new axis and start save array 'keepArray'
        keepArray = wholeNet[..., np.newaxis]
        for netw in networks:
            nowNet = Storage[netw]
            keepArray = np.concatenate((keepArray,
                                        nowNet[..., np.newaxis]), axis=3)

        # save out whole network for testing purposes
        wholeOut = nib.Nifti1Image(keepArray,
                                   networkFile.get_affine(),
                                   networkFile.get_header())

        nib.nifti1.save(wholeOut, 'wholeNetworkSave.nii.gz')

    else:
        print 'node and network File have different dimensions'
        print 'aborting'

    return Storage


if __name__ == '__main__':
    print 'Running in direct Call mode!'

    # fetch the two filenames for passing them to the Main() function
    networkFile = nib.load(sys.argv[1])
    nodeFile = nib.load(sys.argv[2])
    Main(networkFile, nodeFile)
