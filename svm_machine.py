# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:22:09 2012

@author: sebastian
"""

import numpy as np


class super():
    """This is a docstring!"""
    def __init__(self, var1, var2, var3):
        self.var = var1
        self.name = var2
        self.test = var3

    def function(self, invar):
        print invar
        print np.mean(self.test)