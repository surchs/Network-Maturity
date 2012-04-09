# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:08:02 2012

@author: sebastian
"""

# third attempt at getting something to run that actually already worked

# first of all, get the imports done
import os
import sys
import time
import numpy as np
import nibabel as nib
from multiprocessing import Pool


"""
we need a Loader, a HeadGrabber, a Masker and an Averager (possibly this
could also be done by the Masker)
lastly, we need a main(): to call the whole shebang
should need to use consistent variable names

Todo:
    - put dir-exists checks into TexSaver and NifSaver!

    - think about a better way to resetting the variables at the end of loops

    - think about a solution for dataset structures where there is no relative
    path for the functional files (i.e. the functional lies directly in the
    subject directory)

    - all variables in camelcase, all functions with capital first letter

    - print # of current subject with total # of subjects next to it
      (i.e. Running subject 10 of 100)

    - think about whether the flexibility of the script should be tailored
      towards the filenames of batchfile.txt and configfile.txt, towards the
      path where these are located or both. Right now we have flexible
      filenames and inflexible paths (though we also have flexible paths if
      called through import)

v1 3/5/12:
    implemented the rough stuff
v2 3/8/12:
    changes include the ability to run multiple files given by a textfile
v3 3/9/12:
    correlationmatrix
v4 3/12/12:
    support vector machine and cleaning up of the code
v5 3/16/12:
    remove svm again (doesn't make sense in this module) and prepare code for
    running separately for different subjects
v6 3/19/12:
    implement parallelization, cleaned up code
v7 3/29/12:
    imporve modularization by removing the main() function
v8 4/4/12:
    manipulate for spectral clustering - first try
    also changing the saving procedure
"""


def Loader(path, source=''):
    # takes in the path to the source and the source(filename) if specified
    # if no source is specified, this assumes that it is parsed directly
    # with the path
    img = nib.load(os.path.join(path, source))
    data = img.get_data().astype('float64')

    return ((img, data))


def HeadGrab(nifti):
    #takes in a nifti file and returns the head
    header = nifti.get_header()
    affine = nifti.get_affine()

    return ((header, affine))


def BatchReader(line, ages):
    # check if an age has been provided
    subConf = line.strip().split()
    ageStore = 1
    if len(subConf) == 1:
        print ('!You did not give an age for some or all of your subjects, '
               'SVR won\'t be available!')
        ageStore = 0

    sub = subConf[0]
    age = np.fromstring(subConf[1], dtype='float32', sep=' ')
    if ageStore == 1:
        ages = np.append(ages, age)
    else:
        ages = np.array([], dtype='float32')

    return (sub, ages)


def Masker(inData, maskData, maskNo):
    # takes in data of a functional file (possibly also anatomical),
    # the maskData and a maskNo
    # create output file of same dimensions as inData
    maskedOut = np.zeros(inData.shape, dtype='float32')
    # assign values inside the mask to the outfile
    maskedOut[maskData == maskNo] = inData[maskData == maskNo]
    cropMask = inData[maskData == maskNo]

    return ((maskedOut, cropMask))


def Averager(masked, averageArray=np.array([], dtype='float32')):
    # takes in a masked functional volume as an array of dimensions:
    # (voxels in map * timepoints)
    # averages over timepoints and stacks the returning averaged vector into
    # an increasing array if the function is called multiple times
    # if no array is handed over during calling, an empty array is used and
    # output will be identical to the averaged vector

    average = np.mean(masked, axis=0).astype('float32')
    if averageArray.size == 0:
        averageArray = average[np.newaxis, ...]
    else:
        averageArray = np.concatenate((averageArray,
                                       average[np.newaxis, ...]), axis=0)

    return ((average, averageArray))


def Correlater(averageArray, feature=np.array([], dtype='float32')):
    # takes in an array of any dimensions
    # treats the rows (axis=0) as observations and calculates a correlation
    # matrix
    # returns a vector of the lower diagonal of the matrix (created rowise)
    # and stacks vectors into an array if called multiple times

    corr = np.corrcoef(averageArray)
    matMask = np.ones_like(corr)
    # get the lower triangle of the matrix without the diagonal (-1)
    matMask = np.tril(matMask, -1)
    featVec = corr[matMask != 0]

    # append the feature matrix
    if feature.size == 0:
        feature = featVec[np.newaxis, ...]
    else:
        feature = np.concatenate((feature, featVec[np.newaxis, ...]), axis=0)

    return ((corr, featVec, feature))


def NifSaver(inFile, path, hdr, aff, prefix='outNiftiti', suffix='default'):
    # takes an array and writes it to a nifti. needs reasonable
    # affine and header information to do so
    outNifti = nib.Nifti1Image(inFile, aff, header=hdr)
    nib.nifti1.save(outNifti, (path + '/' + prefix + '_' + suffix + '.nii.gz'))


def TexSaver(inFile, path, prefix='avderage', suffix='default'):
    # writes anything to a textfile
    np.save((path + '/' + prefix + '_' + suffix + '.npy'), inFile)


def SubjectProcesser(arguments):

    (sub,
     funcAbsPath,
     funcRelPath,
     funcName,
     maskData) = arguments

    # load the functional file
    (funcSet, funcData) = Loader(os.path.join(funcAbsPath, sub),
                                 source=funcName)
    feature = np.array([], dtype='float32')
    averageArray = np.array([], dtype='float32')

    # for every unique mask value (because we can't assume that the values are
    # are continues from 1 to max_value(mask))
    for num in np.unique(maskData):
        # average the mask that was just run
        (outMask, smallMask) = Masker(funcData, maskData, num)
        (avgerage, averageArray) = Averager(smallMask,
                                            averageArray=averageArray)

    # when the average is built over all nodes in the mask, we calculate the
    # correlation matrix and output it into the variable 'corr' for each
    # subject. The lower triangle of this matrix is the feat vector.
    # the feat vectors for all subjects stacked in the order of processing
    # (or in the order of argument list in multicore-mode) is called 'feature'
    (corr, featVec, feature) = Correlater(averageArray)

    print 'Done with Subject', sub, 'now '
    return (featVec, corr)


def MaskWrapper(maskArguments, maskNo=0):
    # check if the values in the mask are actually continuous from 1 to
    # max_value(mask)
    (subList,
     ageArray,
     funcAbsPath,
     funcName,
     funcRelPath,
     maskData,
     numberProcesses) = maskArguments

    # check if mask is 3D or 4D
    if len(maskData.shape) > 3:
        localMask_ = maskData[..., maskNo]
    else:
        localMask_ = maskData

    maskUnique = np.unique(localMask_)
    maskRange = np.arange(1, int(localMask_.max() + 1))
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

    feature = np.array([], dtype='float32')
    corrStore = np.array([], dtype='float32')
    argList = []

    for sub in subList:
        subArguments = (sub,
                     funcAbsPath,
                     funcRelPath,
                     funcName,
                     localMask_)
        argList.append(subArguments)

    print 'Starting Multicore Processing with', str(numberProcesses), 'cores.'
    start = time.time()
    pool = Pool(processes=numberProcesses)
    resultList = pool.map(SubjectProcesser, argList)
    stop = time.time()
    elapsed = stop - start
    print 'Multicore Processing done. Took', str(elapsed), 'seconds'

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

    return (feature, corrStore)


def Main(batchFile, configFile, sysPath, saveOut=1):
    print '\nWelcome to the nifti Processer! '

    # read from configfile
    conf = open(os.path.join(sysPath, configFile)).readline().strip().split()
    (mPath,
     mSource,
     funcAbsPath,
     funcName,
     funcRelPath,
     outPath) = conf

    # read from batchfile
    batch = open(os.path.join(sysPath, batchFile))
    ages = np.array([], dtype='float32')
    subList = []
    # for storing the looped variables, we are using dictionaries instead of
    # arrays as we cannot know the dimensions of the returned arrays before
    # and stacking arrays with different dimensions is uncool
    featureDict = {}
    correlationDict = {}
    numberProcesses = 8

    for line in batch:
        (sub, ages) = BatchReader(line, ages)
        subList.append(sub)
    ageArray = ages

    # now subList and ageList contain information on the subjects

    (mRaw, maskData) = Loader(mPath, source=mSource)

    # regardless of iteration of loop, all processes get the
    # same base arguments
    maskArguments = (subList,
                     ageArray,
                     funcAbsPath,
                     funcName,
                     funcRelPath,
                     maskData,
                     numberProcesses)

    # insert a switch for maskfiles with more than one mask in them
    # for example a 4D maskfile, where the different networks are stacked
    # along the fourth dimension (we will expect this, stacking along another
    # dimension is currently not implemented)
    if len(maskData.shape) > 3:
        # we are using a 4D mask, let's tell the world
        print '\n##### 4D mask detected ##### '
        print 'You have supplied a 4D mask file '
        print 'Masks are assumed to be stacked along the 4th dimension '
        print ''
        print 'For each network, a separate feature matrix will be generated '
        print 'and stored under a keyword in a dictionary '
        print '##### 4D mask detected #####\n '

        # loop over the masks
        for maskNo in range(maskData.shape[3]):
            (feature, corrStore) = MaskWrapper(maskArguments, maskNo=maskNo)

            # store the returned arrays into the dictionary
            featureDict[('network ' + str(maskNo))] = feature
            correlationDict[('network ' + str(maskNo))] = corrStore

    else:
        # we are using a standard 3D mask, no need for any loops
        print '\nYou are using a 3D mask '
        print 'All is fine with using a 3D mask. It just means that all your '
        print 'computations will be based on only one network / node. '
        print 'Have a nice day! '
        (feature, corrStore) = MaskWrapper(maskArguments)

        featureDict['only network'] = feature
        correlationDict['only network'] = corrStore

    if saveOut == 1:
        print '\n########## '
        print 'Saving data to files '
        # new archive saving method
        prefix = 'full_archive'
        filename = prefix
        fullname = os.path.join(outPath, filename)
        np.savez(fullname,
                 feature=featureDict,
                 correlation=correlationDict,
                 time=time.asctime(),
                 batchfile=batchFile,
                 syspath=sysPath)
        print '\nDone saving! '
        print '########## '
    else:
        print '\n########## '
        print 'Nothing will be saved because \'saveout\' has not been set. '
        print '########## '

    # Return the values if this was called from another module
    return (feature, ages, corrStore)


# Boilerplate to call main():
if __name__ == '__main__':
    print 'Running in direct Call mode!'

    # Information from the configfile should be fetched here so Main() can
    # be cleaned up for modular use

    sysPath = os.path.abspath(os.curdir)
    batchFile = sys.argv[1]
    configFile = sys.argv[2]
    Main(batchFile, configFile, sysPath)