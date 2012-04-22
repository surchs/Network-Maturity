# -*- coding: utf-8 -*-
"""

@author: surchs

mask the fuckers
"""

import os
import re
import numpy as np
import sys
import glob
import nibabel as nib
# import numpy as np

maskPath = sys.argv[1]
dataPath = sys.argv[2]
fileName = sys.argv[3]
outName = sys.argv[4]

mask = nib.load(maskPath)
maskData = mask.get_data()

debug = 0
ran = 0
print dataPath
print ran
for infile in glob.glob(os.path.join(dataPath, ('*/', fileName))):

    print 'current file is', infile
    if debug != 0:
        func = nib.load(infile)
        funcData = func.get_data()
        cleanData = maskData[..., np.newaxis] * funcData
        funcOut = nib.Nifti1Image(cleanData,
                                  func.get_affine(),
                                  func.get_header())
        outPath = re.sub(r'lfo_masked.nii.gz', outName, infile)
        print 'the new output is:', outPath
        nib.nifti1.save(funcOut, outPath)
    else:
        print 'just debugging'