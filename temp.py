# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 18:57:21 2012

@author: sebastian
"""


    # check if the values in the mask are actually continuous from 1 to
    # max_value(mask)
    maskUnique = np.unique(maskData)
    maskRange = np.arange(1, int(maskData.max() + 1))
    matchCheck = np.in1d(maskUnique, maskRange)
    # if there is any value that is missing
    if matchCheck.any(False):
        missing = (matchCheck == False)
        print '\n##### ATTENTION #####'
        print 'Your mask is not continuous!'
        print 'This means that the values in your mask are not continuous from'
        print '1 to the maximum value'
        print 'these values are missing:', maskUnique[missing]
        print '\nIf you expected this, all is fine.'
        print 'If not, you should consider checking your mask'
        print '##### ATTENTION #####\n'

    batch = open(os.path.join(sysPath, batchFile))
    ages = np.array([], dtype='float32')
    argList = []
    texOut = 0
    numberProcesses = 8

    # prepare storage variables
    subjectOrder = []
    # these to put the output of multiple processes into
    feature = np.array([], dtype='float32')
    corrStore = np.array([], dtype='float32')

    for line in batch:
        (sub, ages) = BatchReader(line, ages)
        subjectOrder.append(sub)
        arguments = (sub,
                     funcAbsPath,
                     funcRelPath,
                     funcName,
                     maskData,
                     outPath,
                     texOut)
        argList.append(arguments)

    pool = Pool(processes=numberProcesses)
    resultList = pool.map(Processer, argList)
    print type(resultList)
    print len(resultList)

    TexSaver(resultList,
             outPath,
             prefix='reslist',
             suffix='newall')

    # collect the outputs of the different processes and put them
    # into stacked variables
    for item in range(len(resultList)):
        # expand the outputs in reslist into variables again
        (featVec, corr) = resultList[item]

        # for the feature vectors
        if feature.size == 0:
            feature = featVec[np.newaxis, ...]
        else:
            feature = np.concatenate((feature,
                                      featVec[np.newaxis, ...]),
                                      axis=0)

        # for the correlation matrices
        if corrStore.size == 0:
            corrStore = corr[np.newaxis, ...]
        else:
            corrStore = np.concatenate((corrStore, corr[np.newaxis, ...]),
                                       axis=0)

    return (corrStore, feature)
