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


def FeatureSelection(feature, labels):
    # This implementation is highly suspicious.
    # In other words, I still don't know what I am doing here
    newFeature = svm.LinearSVC(penalty="l2").fit_transform(feature, labels)
    featNo = feature.shape[1]
    newFeatNo = newFeature.shape[1]

    print 'The new number of features is:', newFeatNo
    print 'this is down from:', featNo
    time.sleep(3)
    return (newFeature, featNo, newFeatNo)


def DataSplit(feature, labels):
    # first, reduce the number of features we are using
    newFeature, featNo, newFeatNo = FeatureSelection(feature, labels)
    # first split the data into train and test
    index, mirrorIndex = KFold(newFeature.shape[0], 2)
    trainData = newFeature[index[0]]
    testData = newFeature[index[1]]
    trainLabel = labels[index[0]]
    testLabel = labels[index[1]]

    return (trainData, trainLabel, testData, testLabel, featNo, newFeatNo)


def ParamEst(trainData, trainLabel):
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
        parameters = {'C': np.arange(firstPassC - float(firstPassC) / 2,
                                     firstPassC + float(firstPassC) / 2,
                                     0.1)}
    else:
        print 'The Firstpass C parameter is:', firstPassC
        parameters = {'C': np.arange(firstPassC - 10, firstPassC + 10, 0.1)}

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
    performance = secondTrainModel.score(trainData, trainLabel)
    print 'Training model score is:', performance

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

            # diff = testmodel.predict(testData[pred]) - testLabel[pred]
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
     testLabel,
     featNo,
     newFeatNo) = DataSplit(feature, labels)

    bestC = ParamEst(trainData,
                     trainLabel)

    trainModel = TrainModel(trainData,
                            trainLabel,
                            bestC)

    (trueKeep, predKeep) = TestModel(trainModel,
                                     testData,
                                     testLabel,
                                     cv=1,
                                     bestC=bestC)

    return (trueKeep, predKeep, trainModel, featNo, newFeatNo)


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
    outPath = os.path.join(sysPath, 'output')
    resolution = archive['resolution']
    print resolution

    machineDict = {}
    # enter kind of a docstring in the dictionary
    machineDict['usage'] = ('The network eintries are tuples of length 5'
                            '\nThey are used in the following fashion:'
                            '   0) true labels for the test dataset '
                            '   1) predicted labels for the test dataset '
                            '   2) the saved model used for predicting '
                            '   3) number of original features '
                            '   4) number of features after feature select. ')
    machineDict['subjects'] = subjects
    machineDict['resolution'] = resolution

    # now we can loop over the networks
    for network in networkList:
        feature = archive[network]

        # check if the mask contains more than one node (otherwise this makes
        # pretty much no sense, right)
        if feature.shape[1] > 1:
            print '\n\nRunning network', str(network), 'now. '
            # store the output in the dictionary again
            machineDict[network] = Processer(feature, labels)
            print 'Network', str(network), 'done. '
        else:
            print '\n\n#####  ATTENTION ##### '
            print 'Network', str(network), 'containst only one node. '
            print 'This makes no sense for this type of anlysis so  '
            print 'network', str(network), 'will be skipped. '
            print 'Please check your mask if you didn\'t expect this. '
            print '#####  ATTENTION ##### '
            time.sleep(2)

    machineDict['time'] = time.asctime()
    numberNetworks = len(networkList)
    filename = ('machine_'
                + str(numberNetworks)
                + '_networks_'
                + str(resolution))

    if saveOut == 1:
        outFile = os.path.join(outPath, filename)
        np.savez(outFile, **machineDict)

    else:
        print 'Nothing saved here, parameters are simply returned.'

    return machineDict

if __name__ == '__main__':
    print 'Running in direct Call mode!'
    sysPath = os.path.abspath(os.curdir)
    storedFile = sys.argv[1]
    archive = np.load(storedFile)
    Main(archive, sysPath)