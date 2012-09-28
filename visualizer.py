'''
Created on Sep 28, 2012

@author: sebastian
'''
import sys
import gzip
import cPickle
import numpy as np


def Main(loadFile):
    # get the MAE of the networks in the file and print them in the command
    # line - for now
    f = gzip.open(loadFile, 'rb')
    analysis = cPickle.load(f)
    for network in analysis.networks.keys():
        net = analysis.networks[network]
        tempDiff = net.trueData - net.predictedData
        tempMAE = np.mean(np.abs(tempDiff))
        print 'MAE for', network, 'is:', str(tempMAE)
    pass

if __name__ == '__main__':
    loadFile = sys.argv[1]
    Main(loadFile)
    pass
