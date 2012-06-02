# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:56:56 2012

@author: surchs
make a common mask. creates a mask that removes values without variance
"""

import os
import sys
import glob
import nibabel as nib
import numpy as np
from multiprocessing import Pool


def Processer(infile):
    print 'current file is', infile
    loadfile = nib.load(infile)
    tempdata = loadfile.get_data()
    tempbin = tempdata
    store = np.zeros_like(tempbin[..., 0])

    for loc in np.argwhere(tempbin[..., 0]):
        x = loc[0]
        y = loc[1]
        z = loc[2]
        if np.var(tempbin[x, y, z, :]) > 0:
            store[x, y, z] = 1

    return (store)


def Main(dataPath, fileName, outName, numberCores):
    ran = 0
    debug = 0
    print dataPath

    fileList = []
    for infile in glob.glob(os.path.join(dataPath, '*/', fileName)):
        fileList.append(infile)

    if debug != 1:
        pool = Pool(processes=numberCores)
        resultList = pool.map(Processer, fileList)

    for result in range(len(resultList)):
        store = resultList[result]

        if ran == 0:
            storage = store
            ran = 1
        else:
            storage = storage * store

    if debug != 1:
        loadfile = nib.load(fileList[0])
        outfile = nib.Nifti1Image(storage,
                                  loadfile.get_affine(),
                                  loadfile.get_header())
        nib.nifti1.save(outfile, outName)
    else:
        print 'nuthin saved here, yo'


if __name__ == '__main__':
    print 'Running in direct Call mode!'
    dataPath = sys.argv[1]
    fileName = sys.argv[2]
    outName = sys.argv[3]
    if len(sys.argv) > 4:
        numberCores = int(sys.argv[4])
    else:
        numberCores = 8
    Main(dataPath, fileName, outName, numberCores)
