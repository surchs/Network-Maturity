# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:00:08 2012

@author: sebastian

visualizer script
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


def Main(archive):
    # prepare the stored file for assignment
    print 'Labels in machinefile:', archive.files
    networkList = [key for key in archive.files if 'network' in key]
    networkList.sort()
    print 'Networks in machinefile:', networkList

    plt.figure(1)
    loc = 1
    resolution = archive['resolution']
    print 'resolution is', resolution

    if len(networkList) > 1:

        for network in networkList:
            # this workaround is necessary because of an ugly implementation
            # so far
            allData = archive[network].reshape(-1)
            trueData = allData[0]
            predData = allData[1]
            oldFeatures = allData[3]
            usedFeatures = allData[4]

            # make smooth fitting
            x2 = np.arange(trueData.min() - 1, trueData.max() + 1, .01)
            c = np.polyfit(trueData, predData, 2)
            y2 = np.polyval(c, x2)

            plt.subplot(3, 3, loc)
            loc += 1
            plt.plot(trueData,
                     predData,
                     'kx')
            plt.plot(x2, y2)
            plt.title((network
                       + ' with '
                       + str(usedFeatures)
                       + ' out of '
                       + str(oldFeatures)
                       + ' original features '))

            plt.suptitle('Plot Title', fontsize=12)

    else:
        network = networkList[0]
        allData = archive[network].reshape(-1)
        trueData = allData[0]
        predData = allData[1]
        oldFeatures = allData[3]
        usedFeatures = allData[4]

        # make smooth fitting
        x2 = np.arange(min(trueData) - 1, max(trueData) + 1, .01)
        c = np.polyfit(trueData, predData, 2)
        y2 = np.polyval(c, x2)

        plt.plot(trueData,
                 predData,
                 'kx')
        plt.plot(x2, y2)
        plt.title((network
                   + ' with '
                   + str(usedFeatures)
                   + ' out of '
                   + str(oldFeatures)
                   + ' original features '))

    plt.show()


    plt.show
    print 'got to here'


if __name__ == '__main__':
    print 'Running in direct Call mode!'
    storedFile = sys.argv[1]
    archive = np.load(storedFile)
    Main(archive)