# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:56:56 2012

@author: surchs
make a common mask
"""

import os
import sys
import glob
import nibabel as nib
import numpy as np

dataPath = sys.argv[1]
fileName = sys.argv[2]
outName = sys.argv[3]
maskValue = int(sys.argv[4])

ran = 0
debug = 0
print dataPath

for infile in glob.glob(os.path.join(dataPath, '*/', fileName)):

    print 'current file is', infile
    if debug != 1:
        loadfile = nib.load(infile)
        tempdata = loadfile.get_data()
        tempbin = tempdata
        tempbin[tempbin != maskValue] = 1
        store = np.zeros_like(tempbin[..., 0])

        for dim in range(tempbin.shape[3]):
            store = store + tempbin[..., dim]

        teststore = store
        teststore[teststore != 0] = 1

        if ran == 0:
            storage = teststore
        else:
            storage = storage * teststore

if debug != 1:
    outfile = nib.Nifti1Image(storage,
                              loadfile.get_affine(),
                              loadfile.get_header())
    nib.nifti1.save(outfile, outName)
else:
    print 'nuthin saved here, yo'
