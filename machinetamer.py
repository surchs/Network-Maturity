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
v2 3/30/12:
    - modularized
"""

import numpy as np
from sklearn import svm, grid_search
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut


def DataSplit(feature, labels):
    # first split the data into train and test
    index, mirrorIndex = KFold(feature.shape[0], 2)
    trainData = feature[index[0]]
    testData = feature[index[1]]
    trainLabel = labels[index[0]]
    testLabel = labels[index[1]]

    return (trainData, trainLabel, testData, testLabel)


def ParamEst(feature, labels, trainData, trainLabel):
    # provide the parameters for the first, coarse pass
    parameters = {'C': np.arange(1, 1000, 2)}
    firstGridModel = svm.SVR()

    print 'First pass parameter estimation! '
    firstTrainModel = grid_search.GridSearchCV(firstGridModel,
                                          parameters,
                                          cv=10,
                                          n_jobs=-1,
                                          verbose=1,
                                          pre_dispatch=8)
    firstTrainModel.fit(trainData, trainLabel)
    firstPassC = firstTrainModel.best_estimator_.C
    print 'The Firstpass C parameter is:', firstPassC

    # reuse estimated parameters for second, better pass
    print '\nSecond pass parameter estimation! '
    secondGridModel = svm.SVR()
    parameters = {'C': np.arange(firstPassC - 30, firstPassC + 30, 0.1)}
    secondTrainModel = grid_search.GridSearchCV(secondGridModel,
                                                parameters,
                                                cv=10,
                                                n_jobs=-1,
                                                verbose=1,
                                                pre_dispatch=8)
    bestC = secondTrainModel.best_estimator_.C
    print 'Overall best C parameter is:', bestC

    return bestC


def TrainModel(trainData, trainLabel, bestC, savemodel=0):
    # Modelparameters from Estimation
    trainModel = svm.SVR(C=bestC)
    trainModel.fit(trainData, trainLabel)

    # This is not used by default because I haven't figured out how to save
    # this to a useful path without pushing the path all the way through
    # the script. Instead this model is returned to the wrapper and saved
    # there.
    if savemodel == 1:
        np.save('trainmodel.npy', trainModel)

    return trainModel


def TestModel(trainModel, testData, testLabel, cv=1, bestC=1):
    # got the model from TrainModel

    if cv == 1:
        # do crossvalidation
        testmodel = svm.SVR(C=bestC)

        # split the dataset for crossvalidation
        loo = LeaveOneOut(len(testLabel))

        trueKeep = np.array([])
        predKeep = np.array([])

        for train, pred in loo:
            testmodel.fit(testData[train], testLabel[train])

            print ('Predicted:',
                   testmodel.predict(testData[pred]),
                   'true',
                   testLabel[pred])

            diff = testmodel.predict(testData[pred]) - testLabel[pred]
            print 'Difference is', diff

            trueKeep = np.append(trueKeep, testLabel[pred])
            predKeep = np.append(predKeep, testmodel.predict(testData[pred]))

    else:
        # don't crossvalidate, just predict
        trueKeep = testLabel
        predKeep = trainModel.predict(testLabel)

    return (trueKeep, predKeep)


def Processer(feature, labels):
    # This is an internal wrapper function to make calling the functions
    # inside this module less of a hassle

    (trainData,
     trainLabel,
     testData,
     testLabel) = DataSplit(feature, labels)

    bestC = ParamEst(feature,
                     labels,
                     trainData,
                     trainLabel)

    trainModel = TrainModel(trainData,
                            trainLabel,
                            bestC)

    (trueKeep, predKeep) = TestModel(trainModel,
                                     testData,
                                     testLabel,
                                     cv=1,
                                     bestC=bestC)

    return (trueKeep, predKeep, trainModel)