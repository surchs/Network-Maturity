# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:01:50 2012

@author: sebastian
"""


    '''elif featNo < 2000 and featNo > 1000:
        rfeObject = RFE(estimator=svrEstimator,
                        n_features_to_select=1000,
                        step=0.01)

        rfeObject.fit(feature, labels)
        reducedFeatureIndex = rfeObject.support_
        reducedFeatures = feature[..., reducedFeatureIndex]

        rfecvObject = RFECV(estimator=svrEstimator,
                            step=2,
                            cv=5,
                            loss_func=mean_squared_error)

        rfecvObject.fit(reducedFeatures, labels)
        featureIndex = rfecvObject.support_
        featureScore = rfecvObject.cv_scores_
        featureStep = rfecvObject.step
        newFeature = reducedFeatures[..., featureIndex] '''