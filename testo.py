# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:56:34 2012

@author: sebastian
"""
import sys


def Main(a, b='hallo'):
    print a
    print b

if __name__ == '__main__':
    print 'Running in direct Call mode!'

    # Information from the configfile should be fetched here so Main() can
    # be cleaned up for modular use

    a = sys.argv[1]
    if len(sys.argv) > 2:
        b = sys.argv[2]
    else:
        b = 'haha'
    Main(a, b=b)