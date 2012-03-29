# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:20:37 2012

@author: sebastian

This is a SVM implementation module
Here is a list of things that the module is supposed to do:

    - based on a given feature and label set, train a model and return
    the model performance

    - automatically run a crossvalidation on the model without the need
    for the user to define crossvalidation groups

    - do the crossvalidation in such a way that the groups are drawn
    randomly (and automatically)

    - estimate the model parameters using a grid search to determine
    the best parameters for model performance

    - implement a visualization of the different iterations

v1 3/21/12:
    - first implementation
"""

import numpy as np
from sklearn import svm, grid_search
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut


def svmer(feature, labels):

    kFoldIndex = KFold(len(labels), 2)
    trainIndex, testIndex = kFoldIndex
    trainSet1 = feature[trainIndex[0]]
    trainSet2 = feature[trainIndex[1]]
    testSet1 = labels[trainIndex[0]]
    testSet2 = labels[trainIndex[1]]

    parameters = {'C': np.arange(180, 240, 0.1)}
    gridModel = svm.SVR()

    trainModel = grid_search.GridSearchCV(gridModel, parameters, cv=10, n_jobs=8)
    trainModel.fit(trainSet1, testSet1)
    bestC = trainModel.best_estimator_.C

    loo = LeaveOneOut(len(testSet2))
    testmodel = svm.SVR(C=bestC)

    trueKeep = np.array([])
    predKeep = np.array([])

    for train, test in loo:
        testmodel.fit(trainSet2[train], testSet2[train])
        print 'Predicted:', testmodel.predict(trainSet2[test]), 'true', testSet2[test]
        diff = testmodel.predict(trainSet2[test]) - testSet2[test]
        print 'Difference is', diff

        trueKeep = np.append(trueKeep, testSet2[test])
        predKeep = np.append(predKeep, testmodel.predict(trainSet2[test]))

    return (trueKeep, predKeep)
