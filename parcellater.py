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


def Averager(corrStore):
    meanCorrMat = np.mean(corrStore, axis=0)


    return


def Parcellater(avCorrMat):

    return


def Processer():

    return


# Boilerplate to call main():
if __name__ == '__main__':
    print 'Running in direct Call mode!'

    # Information from the configfile should be fetched here so Main() can
    # be cleaned up for modular use

    sysPath = os.path.abspath(os.curdir)
    batchFile = sys.argv[1]
    configFile = sys.argv[2]
    Main(batchFile, configFile, sysPath)