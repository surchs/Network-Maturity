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
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.feature_selection import RFE, RFECV


def FeatureSelection(feature, labels):
    # Now this implementation is becoming less suspicious because I feel
    # that I start to understand what I am doing here, at least for the
    # linear kernel
    #
    # first: let's get the current number of features
    featNo = feature.shape[1]
    # we also need an estimator object for scoring during RFE
    svrEstimator = svm.SVR(kernel='linear')
    # if there are too many features, we cannot run recursive feature
    # elimination with performance measures. We first have to reduce the
    # number of features the hard way
    # here we set the threshold to 3000 - maybe too much...
    if featNo > 1000:
        rfeObject = RFE(estimator=svrEstimator,
                        n_features_to_select=1000,
                        step=0.01)

        rfeObject.fit(feature, labels)
        reducedFeatureIndex = rfeObject.support_
        reducedFeatures = feature[..., reducedFeatureIndex]

        rfecvObject = RFECV(estimator=svrEstimator,
                            step=1,
                            cv=5,
                            loss_func=mean_squared_error)

        rfecvObject.fit(reducedFeatures, labels)
        newfeatureIndex = rfecvObject.support_
        featureScore = rfecvObject.cv_scores_
        featureStep = rfecvObject.step
        newFeature = reducedFeatures[..., newfeatureIndex]

    else:
        rfecvObject = RFECV(estimator=svrEstimator,
                            step=1,
                            cv=5,
                            loss_func=mean_squared_error)

        rfecvObject.fit(feature, labels)
        featureIndex = rfecvObject.support_
        featureScore = rfecvObject.cv_scores_
        featureStep = rfecvObject.step
        newFeature = feature[..., featureIndex]

    newFeatNo = newFeature.shape[1]

    return (newFeature, featNo, newFeatNo, featureScore, featureStep)


def DataSplit(feature, labels):
    # first, reduce the number of features we are using
    print 'Starting feature selection '
    (newFeature,
     featNo,
     newFeatNo,
     featureScore,
     featureStep) = FeatureSelection(feature, labels)
    print 'Feature selection is done'
    # first split the data into train and test
    index, mirrorIndex = KFold(newFeature.shape[0], 2)
    trainData = newFeature[index[0]]
    testData = newFeature[index[1]]
    trainLabel = labels[index[0]]
    testLabel = labels[index[1]]

    return (trainData, trainLabel, testData, testLabel, featNo, newFeatNo)


def ParamEst(trainData, trainLabel, numberCores):
    # switch if the number of samples is less than 10
    # otherwise the crossvalidation will fail
    # first set the default cv value:
    cv = 10
    if len(trainLabel) < cv:
        cv = len(trainLabel)

    # provide the parameters for the first, coarse pass
    parameters = {'C': np.arange(1, 1000, 2).tolist()}
    # print 'Parameters', parameters['C']
    gridModel = svm.SVR(kernel='linear')

    print 'First pass parameter estimation! '
    firstTrainModel = grid_search.GridSearchCV(gridModel,
                                               parameters,
                                               cv=cv,
                                               n_jobs=numberCores,
                                               pre_dispatch='2*n_jobs',
                                               verbose=1)
    firstTrainModel.fit(trainData, trainLabel)
    firstPassC = firstTrainModel.best_estimator_.C
    firstPassE = firstTrainModel.best_estimator_.epsilon

    if firstPassC < 32:
        print 'The Firstpass C parameter is less than 30 '
        print 'Because of the current implementation, this requires a '
        print 'different way of parameter generation '
        print 'This is just to let you know that this happened. '
        print 'The Firstpass C parameter is', firstPassC, 'by the way'
        print 'and epsilon is', firstPassE
        parameters = {'C': np.arange(firstPassC - float(firstPassC) / 2,
                                     firstPassC + float(firstPassC) / 2,
                                     0.1)}
    else:
        print 'The Firstpass C parameter is:', firstPassC
        print 'and epsilon is', firstPassE
        parameters = {'C': np.arange(firstPassC - 10, firstPassC + 10, 0.1)}

    # reuse estimated parameters for second, better pass
    print '\nSecond pass parameter estimation! '

    secondTrainModel = grid_search.GridSearchCV(gridModel,
                                                parameters,
                                                cv=cv,
                                                n_jobs=numberCores,
                                                pre_dispatch='2*n_jobs',
                                                verbose=1)
    secondTrainModel.fit(trainData, trainLabel)
    bestC = secondTrainModel.best_estimator_.C
    bestE = secondTrainModel.best_estimator_.epsilon
    print 'Overall best C parameter is:', bestC
    print 'Overall best E parameter is:', bestE

    return (bestC, bestE)


def TrainModel(trainData, trainLabel, bestC, bestE, savemodel=0):
    # Modelparameters from Estimation
    trainModel = svm.SVR(kernel='linear', C=bestC)
    trainModel.fit(trainData, trainLabel)

    # This is not used by default because I haven't figured out how to save
    # this to a useful path without pushing the path all the way through
    # the script. Instead this model is returned to the wrapper and saved
    # there.
    if savemodel == 1:
        np.save('trainmodel.npy', trainModel)

    return trainModel


def TestModel(trainModel, testData, testLabel, cv=1, bestC=1, bestE=0.1):
    # got the model from TrainModel

    if cv == 1:
        # do crossvalidation
        testmodel = svm.SVR(kernel='linear', C=bestC)

        # split the dataset for crossvalidation
        loo = LeaveOneOut(len(testLabel))

        trueKeep = np.array([])
        predKeep = np.array([])
        performance = np.array([])

        for train, pred in loo:
            testmodel.fit(testData[train], testLabel[train])

            # print ('Predicted:',
            #       testmodel.predict(testData[pred]),
            #       'true',
            #       testLabel[pred])

            # diff = testmodel.predict(testData[pred]) - testLabel[pred]
            # print 'Difference is', diff
            trueVal = testLabel[pred]
            predVal = testmodel.predict(testData[pred])

            trueKeep = np.append(trueKeep, trueVal)
            predKeep = np.append(predKeep, predVal)
            se = np.square(trueVal - predVal)
            performance = np.append(performance, se)

    else:
        # don't crossvalidate, just predict
        trueKeep = testLabel
        predKeep = trainModel.predict(testData)
        performance = np.square(testLabel - trainModel.predict(testData))

    rmse = np.mean(np.sqrt(performance))
    print 'mean RMSE =', rmse

    return (trueKeep, predKeep, rmse)


def Processer(trainData,
              trainLabel,
              testData,
              testLabel,
              featNo,
              newFeatNo,
              numberCores,
              estParameters,
              givenC,
              givenE):

    if estParameters == 1:

        bestC, bestE = ParamEst(trainData,
                                trainLabel,
                                numberCores)

    else:
        # announce this
        print '\n##### Attention ##### '
        print 'Parameters were not estimated. '
        print 'instead these given parameters were used: '
        print '     C =', givenC
        print '     E =', givenE
        print '##### Attention ##### \n '
        bestC = givenC
        bestE = givenE
        time.sleep(3)

    trainModel = TrainModel(trainData,
                            trainLabel,
                            bestC,
                            bestE)

    (trueKeep, predKeep, rmse) = TestModel(trainModel,
                                           testData,
                                           testLabel,
                                           cv=0,
                                           bestC=bestC,
                                           bestE=bestE)

    return (trueKeep, predKeep, trainModel, featNo, newFeatNo, rmse)


def Main(preprocDict,
         conditionName,
         outPath,
         numberCores=8,
         estParameters=1,
         givenC=1,
         givenE=0.1,
         saveFiles=True):
    # first get all the unique
    # labels for the features
    labels = preprocDict['labels']

    # list of subject names
    subjects = preprocDict['subjects']

    # next get all the networks inside the file
    networkList = [key for key in preprocDict.keys() if 'network' in key]
    networkList.sort()
    machineDict = {}

    # enter kind of a docstring in the dictionary
    machineDict['usage'] = ('The network eintries are tuples of length 5'
                            '\nThey are used in the following fashion:'
                            '   0) true labels for the test dataset '
                            '   1) predicted labels for the test dataset '
                            '   2) the saved model used for predicting '
                            '   3) number of original features '
                            '   4) number of features after feature select. '
                            '   5) model performance on the test data. ')
    machineDict['subjects'] = subjects

    # now we can loop over the networks
    for network in networkList:
        feature = preprocDict[network]

        # check if the mask contains more than one node (otherwise this makes
        # pretty much no sense, right)
        if feature.shape[1] > 1:
            print '\n\nRunning network', str(network), 'now. '
            (trainData,
             trainLabel,
             testData,
             testLabel,
             featNo,
             newFeatNo) = DataSplit(feature, labels)

            if newFeatNo > 2:
                print 'The new number of features is:', newFeatNo
                print 'this is down from:', featNo
                print '>>Starting network:', str(network)
                time.sleep(3)

                # store the output in the dictionary again
                machineDict[network] = Processer(trainData,
                                                 trainLabel,
                                                 testData,
                                                 testLabel,
                                                 featNo,
                                                 newFeatNo,
                                                 numberCores,
                                                 estParameters,
                                                 givenC,
                                                 givenE)
                print 'Network', str(network), 'done. '
            else:
                print '##### Too few features ##### '
                print 'The new number of features would be:', newFeatNo
                print 'This makes no sense, so network', str(network)
                print 'will be skipped '
                print '##### Too few features ##### '
                time.sleep(5)
        else:
            print '\n\n#####  ATTENTION ##### '
            print 'Network', str(network), 'containst only one node. '
            print 'This makes no sense for this type of anlysis so  '
            print 'network', str(network), 'will be skipped. '
            print 'Please check your mask if you didn\'t expect this. '
            print '#####  ATTENTION ##### '
            time.sleep(2)

    machineDict['time'] = time.asctime()
    filename = ('machine_' + conditionName)

    if saveFiles == 1:
        outFile = os.path.join(outPath, filename)
        np.savez(outFile, **machineDict)

    else:
        print 'Nothing saved here, parameters are simply returned.'

    return machineDict


if __name__ == '__main__':
    print 'Running in direct Call mode!'
    storedFile = sys.argv[1]
    preprocDict = np.load(storedFile)
    conditionName = sys.argv[2]
    outPath = sys.argv[3]

    if len(sys.argv) > 4:
        numProc = int(sys.argv[4])
        Main(preprocDict, conditionName, outPath, numberCores=numProc)
    else:
        Main(preprocDict, conditionName, outPath)