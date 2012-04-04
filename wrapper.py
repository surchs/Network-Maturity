# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:58:22 2012

@author: sebastian

wrapper script
"""

import os
import sys
import preprocesser_v7 as pp
import machinetamer as mt

sysPath = os.path.abspath(os.curdir)
batchFile = sys.argv[1]
configFile = sys.argv[2]

print 'Welcome to wrapper '
print 'All variables have been read in '
print 'The Syspath is', sysPath
print 'Let\'s go '

print '\nStarting the Preprocessing '
(feature, ages, corrStore) = pp.Main(batchFile,
                              configFile,
                              sysPath,
                              saveOut=1)

print '\nStarting SVM '
(trueKeep, predKeep, trainModel) = mt.Processer(feature,
                                                ages)
print 'Done with SVM\n '

print 'Saving TrueKeep '
pp.TexSaver(trueKeep,
            (sysPath + '/output'),
            prefix='trueVals',
            suffix='temp')

print 'Saving PredKeep '
pp.TexSaver(predKeep,
            (sysPath + '/output'),
            prefix='predVals',
            suffix='temp')

print 'Saving TrainModel '
pp.TexSaver(trainModel,
            (sysPath + '/output'),
            prefix='trainModel',
            suffix='temp')