# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 22:43:19 2012

@author: sebastian
"""

import sys
import time


tempvar1 = 'batchlala'
tempvar2 = 'conflala'
comstring = ('This is the comstring'
             '\nHere we go: '
             'Command 1: '
             + tempvar1
             + ' Command 2: '
             + tempvar2
             + '\nTime: '
             + time.asctime())

print comstring
print sys.argv[0]