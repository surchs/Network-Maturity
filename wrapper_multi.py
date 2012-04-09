# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:58:22 2012

@author: sebastian

wrapper script
"""

import os
import sys
import time
import preprocesser_multi as pp
import machinetamer_multi as mt

sysPath = os.path.abspath(os.curdir)
batchFile = sys.argv[1]
configFile = sys.argv[2]

print 'Welcome to wrapper '
print 'All variables have been read in '
print 'The Syspath is', sysPath
print 'Let\'s go '

print '\nStarting the Preprocessing '
ppStart = time.time()
storageDict = pp.Main(batchFile,
                      configFile,
                      sysPath,
                      saveOut=1)
print 'Done with Preprocessing '
ppStop = time.time()
ppElapsed = ppStart - ppStop
print 'Took', ppElapsed, 'seconds to complete'

print '\nStarting SVM '
mtStart = time.time()
machineDict = mt.Main(storageDict,
                      sysPath)
print 'Done with SVM\n '
mtStop = time.time()
mtElapsed = ppStart - ppStop
print 'Took', mtElapsed, 'seconds to complete'