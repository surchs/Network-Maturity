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

import os
import sys
import time
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
    parameters = {'C': np.arange(1, 1000, 2).tolist()}
    # print 'Parameters', parameters['C']
    gridModel = svm.SVR()

    print 'First pass parameter estimation! '
    firstTrainModel = grid_search.GridSearchCV(gridModel,
                                               parameters,
                                               cv=10,
                                               n_jobs=8,
                                               verbose=1)
    firstTrainModel.fit(trainData, trainLabel)
    firstPassC = firstTrainModel.best_estimator_.C

    if firstPassC < 32:
        print 'The Firstpass C parameter is less than 30 '
        print 'Because of the current implementation, this requires a '
        print 'different way of parameter generation '
        print 'This is just to let you know that this happened. '
        print 'The Firstpass C parameter is', firstPassC, 'by the way'
        parameters = {'C': np.arange(firstPassC - firstPassC / 2,
                                     firstPassC + firstPassC / 2,
                                     0.1)}
    else:
        print 'The Firstpass C parameter is:', firstPassC
        parameters = {'C': np.arange(firstPassC - 30, firstPassC + 30, 0.1)}

    # reuse estimated parameters for second, better pass
    print '\nSecond pass parameter estimation! '

    secondTrainModel = grid_search.GridSearchCV(gridModel,
                                                parameters,
                                                cv=10,
                                                n_jobs=8,
                                                verbose=1)
    secondTrainModel.fit(trainData, trainLabel)
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

            # print ('Predicted:',
            #       testmodel.predict(testData[pred]),
            #       'true',
            #       testLabel[pred])

            diff = testmodel.predict(testData[pred]) - testLabel[pred]
            # print 'Difference is', diff

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


def Main(archive, sysPath):
    # first get all the unique
    # labels for the features
    labels = archive['labels']

    # list of subject names
    subjects = archive['subjects']

    # next get all the networks inside the file
    networkList = [key for key in archive.files if 'network' in key]
    networkList.sort()

    # these two parameters could be dynamically assigned for greater usability
    saveOut = 1
    outpath = os.path.join(sysPath, 'output')

    machineDict = {}
    # enter kind of a docstring in the dictionary
    machineDict['usage'] = ('The network eintries are tuples of length 3'
                            '\nThey are used in the following fashion:'
                            '   1) true labels for the test dataset '
                            '   2) predicted labels for the test dataset '
                            '   3) the saved model used for predicting ')
    machineDict['subjects'] = subjects

    # now we can loop over the networks
    for network in networkList:
        feature = archive[network]
        # store the output in the dictionary again
        print '\n\nRunning network', str(network), 'now. '
        machineDict[network] = Processer(feature, labels)
        print 'Network', str(network), 'done. '

    machineDict['time'] = time.asctime()

    if saveOut == 1:
        np.savez(os.join.path(outpath, 'machine_file'), **machineDict)

    else:
        print 'Nothing saved here, parameters are simply returned.'

    return machineDict

if __name__ == '__main__':
    print 'Running in direct Call mode!'
    sysPath = os.path.abspath(os.curdir)
    storedFile = sys.argv[1]
    archive = np.load(storedFile)
    Main(archive, sysPath)