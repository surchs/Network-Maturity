# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:58:26 2012

@author: sebastian

classes for performing the Network based maturity Preproc

"""
### Imports ###
import os
import re
# import sys
import time
import gzip
import cPickle
# import thread
import shutil
import numpy as np
import nibabel as nib
from scipy import stats as st
import multiprocessing as mp
from sklearn import svm, grid_search
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import LeaveOneOut
from sklearn.feature_selection import RFE, RFECV


### Data handling ###
def storeData(dataObject, outPath, outName):
    # this is a function to store analysis or data objects on harddisk
    # here we don't have to check if it is analysis or preprocess because
    # this has been done with the outPath
    # just saving
    tempName = ('temp_' + outName)
    tempFilePath = os.path.join(outPath, tempName)
    tempOutFile = gzip.open(tempFilePath, 'wb')
    cPickle.dump(dataObject, tempOutFile, 2)
    tempOutFile.close()
    # now we get it in the real file
    filePath = os.path.join(outPath, outName)
    shutil.copy2(tempFilePath, filePath)
    # and get rid of the tempFile
    os.remove(tempFilePath)
    print '\n#### Done saving', dataObject.name, '####\n'


### Error handling ###
class MaskError(Exception):
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return repr(self.text)


### Service Objects ###
class Parameters(object):

    def __init__(self, name):
        # here, we just create the object and give it a name
        self.name = name

    def makeStudy(self,
                  configPath,
                  outPath,
                  numberCores):

        self.configPath = configPath
        self.outPath = outPath
        self.numberCores = numberCores

    def makePreproc(self,
                    maskDir,
                    maskName,
                    funcMaskName,
                    funcDir,
                    funcRelPath,
                    funcName,
                    subjectList,
                    outputPath,
                    useFuncMask,
                    nodeMode,
                    cv,
                    fs,
                    kernel,
                    estParameters,
                    givenC,
                    givenE,
                    conditionName,
                    numberCores):
        # define all the Parameters
        self.maskDir = maskDir
        self.maskName = maskName
        self.funcMaskName = funcMaskName
        self.funcDir = funcDir
        self.funcRelPath = funcRelPath
        self.funcName = funcName
        self.subjectList = subjectList
        self.outputPath = outputPath
        self.useFuncMask = useFuncMask
        self.nodeMode = nodeMode
        self.cv = cv
        self.fs = fs
        self.kernel = kernel
        self.estParameters = estParameters
        self.givenC = givenC
        self.givenE = givenE
        self.conditionName = conditionName
        self.numberCores = numberCores

    def makeSubject(self,
                    subjectPath,
                    age,
                    networks,
                    useFuncMask,
                    funcMaskData,
                    nodeMode):
        # method to generate Subject Parameters
        self.subjectPath = subjectPath
        self.age = age
        self.networks = networks
        self.useFuncMask = useFuncMask
        self.funcMaskData = funcMaskData
        self.nodeMode = nodeMode

    def makeNetwork(self):
        pass

    def introduce(self):
        # method for reporting the current Parameters of the object
        print 'Hi, my name is', self.name
        print 'my Parameters are:', self.__dict__


class Storage(object):
    # a class to store information that is too complex for dictionaries
    def __init__(self, name):
        self.name = name

    def putNetwork(self,
                   maskData):

        self.type = 'Network'
        self.maskData = maskData

        # initiating attributes that can be filled later
        self.timeseries = None
        self.nodeVector = None
        self.index = None
        self.networkVector = None
        self.connectivity = {}
        self.features = {}

    def putSubject(self,
                   age,
                   feature):
        # we use these to keep the subjects somewhere for the machine learning
        # analysis
        self.type = 'Subject'
        self.age = age
        self.feature = feature

### Data objects ###


class Study(object):
    # class for the Study object that contains all the preproc and the
    # list of Parameters to control these preproc
    def __init__(self, Parameters):

        self.name = Parameters.name
        self.configPath = Parameters.configPath
        self.outPath = Parameters.outPath
        self.numberCores = Parameters.numberCores

        # initialize the parameter dictionary for the preprocessing
        # because its a pain to write 'preprocessing' everytime, we'll just
        # call it 'preproc' here
        self.preprocParameters = {}
        self.preproc = {}
        self.preprocPaths = {}
        # and the same stuff for the machine
        self.analysisParameters = {}
        self.analyses = {}
        self.analysesPaths = {}

    def getParameters(self):
        # method to read in the config file and put values in attributes
        #
        # right now, the attribute order is hardwired. I could change this
        # to a more flexible order in the future
        configFile = open(self.configPath)
        configLines = configFile.readlines()

        # loop over the lines
        for line in configLines:
            # check if line is commented out
            print '\nChecking the commands on correctness'
            print 'I got', len(line.strip().split()), 'commands'
            if '#' in line or re.match('^\n', line):
                print 'Not using this:', line.strip()
            # check if right number of commands are parsed
            elif len(line.strip().split()) != 17:
                print '\nWrong number of commands '
            else:
                print '\nCommand OK! '
                commandString = line.strip().split()
                maskDir = commandString[0]
                maskName = commandString[1]
                funcMaskName = commandString[2]
                funcDir = commandString[3]
                # funcRelPath = commandString[4]
                funcName = commandString[5]
                subjectList = commandString[6]
                outputPath = commandString[7]
                useFuncMask = int(commandString[8])
                cv = int(commandString[9])
                fs = int(commandString[10])
                kernel = commandString[11]
                nodeMode = int(commandString[12])
                estParameters = int(commandString[13])
                givenC = float(commandString[14])
                givenE = float(commandString[15])
                conditionName = commandString[16]

                # initialize the parameter object inside a temporary variable
                tempObj = Parameters(conditionName)
                # and fill in the Parameters that were just read in
                tempObj.makePreproc(maskDir=maskDir,
                                     maskName=maskName,
                                     funcMaskName=funcMaskName,
                                     funcDir=funcDir,
                                     funcRelPath='',
                                     funcName=funcName,
                                     subjectList=subjectList,
                                     outputPath=outputPath,
                                     useFuncMask=useFuncMask,
                                     nodeMode=nodeMode,
                                     cv=cv,
                                     fs=fs,
                                     kernel=kernel,
                                     estParameters=estParameters,
                                     givenC=givenC,
                                     givenE=givenE,
                                     conditionName=conditionName,
                                     numberCores=self.numberCores)
                # store the complete parameter object in the dictionary
                self.preprocParameters[conditionName] = tempObj

    def makePreproc(self):
        # method to loop over the dictionary with parameter objects and
        # create the corresponding Preproc objects
        #
        # the process is rather simple, just create a new instance of the
        # Preproc class and hand over the parameter object
        for preproc in self.preprocParameters.keys():
            # get the respective parameter object from the dictionary
            tempParameter = self.preprocParameters[preproc]
            print 'temporary Parameters'
            print tempParameter.subjectList
            # put the Preproc object in a temporary object
            print type(Preproc)
            tempObj = Preproc(tempParameter)
            # and save it to the preprocesses dictionary
            self.preproc[preproc] = tempObj

    def initializePreproc(self):
        # Loop over all the stored preprocesses and run the initialization
        for preproc in self.preproc.keys():
            # get the Preproc object
            tempObj = self.preproc[preproc]
            # run the initialization
            tempObj.getPaths()
            # create the masks inside the Preproc
            tempObj.makeMask()
            # and create subjects
            tempObj.makeSubjectParameters()
            tempObj.makeSubject()

    def givePreproc(self):
        # method to hand over the dictionary with preproc to the external
        # wrapper
        return self.preproc

    def runPreproc(self):
        # this method is essentially not used as it is implemented
        # in the wrapper at the moment
        # Loop over all the stored preproc and run the preprocessing
        for preprocName in self.preproc.keys():
            # get the Preproc object
            tempObj = self.preproc[preprocName]
            # run the preprocessing
            tempObj.runPreprocessing()
            self.saveThis('Preprocessing', preprocName)

    def saveThis(self, what, name):
        # method to save either preprocessing or analysis
        #
        # switch
        if what == 'Preprocessing':
            # save preprocessing
            preprocObject = self.preproc[name]
            # generate path
            outPath = self.outPath
            outName = ('preproc_' + preprocObject.name + '.gz')
            outFilePath = os.path.join(outPath, outName)
            self.preprocPaths[name] = outFilePath
            # run the dude
            print ('\n### Saving###' + '\nSaving '
                   + preprocObject.name
                   + ' to '
                   + outFilePath)
            p = mp.Process(target=storeData, args=(preprocObject,
                                                   outPath,
                                                   outName))
            p.start()
            # and now get rid of the preproc object in memory
            self.preproc[name] = None
            pass

        elif what == 'Analysis':
            # save analysis
            analysisObject = self.analyses[name]
            # generate path
            outPath = self.outPath
            outName = ('analysis_' + analysisObject.name + '.gz')
            outFilePath = os.path.join(outPath, outName)
            # store the outPath in the appropriate dictionary
            self.analysesPaths[name] = outFilePath
            # run the dude
            print ('\n### Saving###' + '\nSaving '
                   + analysisObject.name
                   + ' to '
                   + outFilePath)
            p = mp.Process(target=storeData, args=(analysisObject,
                                                   outPath,
                                                   outName))
            p.start()
            # and now get rid of the analysisObject in memory

            self.analyses[name] = None
            pass

        else:
            # don't save anything
            print 'Not saving anything, wrong pointer given'
            pass

    def loadThis(self, what, name):
        # method to save either preprocessing or analysis
        #
        # switch
        if what != 'Preprocessing' and what != 'Analysis':
            # don't save anything
            print 'Not loading anything, wrong pointer given'

        else:
            # we can do something
            if what == 'Preprocessing':
                openPath = self.preprocPaths[name]

            elif what == 'Analysis':
                openPath = self.analysesPaths[name]

            # now check it the file exists:
            if not os.path.exists(openPath):
                # probably the cPickle is still saving, so we have to wait
                print ('\n\n#### File ' + openPath + ' is missing ####'
                       + '\nUnfortunately we will have to wait for the file')

                # set the start point to keep track
                start = time.time()
                while not os.path.exists(openPath):
                    time.sleep(3)
                    stop = time.time()
                    elapsed = stop - start
                    print 'Waited', str(elapsed), 'seconds'
                print 'Yeah, file came up!'
                openFile = gzip.open(openPath)
            else:
                # file exists, we can start right away
                print 'get going'
                openFile = gzip.open(openPath)
            # and do your thing
            dataObject = cPickle.load(openFile)
            return dataObject

    def initializeAnalysis(self):
        # method to run the machine
        # probably the preproc files do no longer exist so they have to be
        # loaded into memory!
        for preprocName in self.preproc.keys():
            # switch
            if self.preproc[preprocName] == None:
                # we need to load the object
                print preprocName, 'no longer in memory, need to load'
                preproc = self.loadThis('Preprocessing', preprocName)
                print preprocName, 'was loaded successfully'
            else:
                # we don't need to load the object, use it from memory
                # create a machine object
                print preprocName, 'still exists in memory, no need to load'
                preproc = self.preproc[preprocName]

            # initialize Analysis object
            tempAnalysis = Analysis(preprocName)
            tempAnalysis.nodeMode = preproc.nodeMode
            tempAnalysis.cv = preproc.cv
            tempAnalysis.fs = preproc.fs
            tempAnalysis.estParameters = preproc.estParameters
            tempAnalysis.kernel = preproc.kernel

            for networkName in preproc.networkNames:
                print networkName
                # if networkName == 'whole brain':
                if False:
                    # do nothing - temporarily shutoff
                    pass
                else:
                    pass
                # prepare the data for the machine
                tempDict = {}
                for subjectName in preproc.subjects.keys():
                    tempSubject = preproc.subjects[subjectName]
                    tempAge = tempSubject.age
                    # feature assignment still hardcoded - meaning that there
                    # is currently no way to choose what kind of connectivity
                    # will be used in the analysis (whole, between, within)
                    tempNetwork = tempSubject.networks[networkName]
                    # at this point, we can choose the hardcoded connectivity
                    tempFeature = tempNetwork.features['within']
                    subject = Storage(subjectName)
                    subject.putSubject(tempAge, tempFeature)
                    tempDict[subjectName] = subject
                # make a new machine object
                network = Network(networkName,
                                  tempDict,
                                  tempAnalysis.cv,
                                  tempAnalysis.fs,
                                  tempAnalysis.kernel,
                                  tempAnalysis.estParameters,
                                  preproc.givenC,
                                  preproc.givenE,
                                  self.numberCores)
                tempAnalysis.networks[networkName] = network
            # store the machines
            self.analyses[preprocName] = tempAnalysis
            # and get rid of the preproc stuff please
            self.preproc[preprocName] = None

    def runAnalysis(self):
        # method to run all the analyses
        for analysisName in self.analyses.keys():
            analysis = self.analyses[analysisName]
            # call the run method of the object
            print '\nRunning analysis', analysisName, '\n'
            analysis.run()
            # when it is finished, store the file and get rid of it in memory
            self.saveThis('Analysis', analysisName)
        print 'Done with the machines'

    def cleanup(self, which):
        # method to force clean every analysis
        if which == 1:
            # we want to clean the preprocessed files
            for preprocName in self.preproc.keys():
                print 'cleaning', preprocName
                self.preproc[preprocName].cleanup()
        elif which == 2:
            # we want to clean the analysed data
            for analysisName in self.analyses.keys():
                print 'cleaning', analysisName
                self.analyses[analysisName].cleanup()

    def saveYourself(self):
        # method to save the current instance of the Study object as a safety
        # measure
        saveFile = gzip.open(os.path.join(self.outPath,
                                          (self.name + '_save.gz')), 'wb')
        cPickle.dump(self, saveFile, protocol=2)


class Preproc(object):
    # class for Preproc objects that contain Network and Subject objects
    # and are themselves stored in the Study object
    def __init__(self, Parameters):

        self.name = Parameters.name
        self.maskDir = Parameters.maskDir
        self.maskName = Parameters.maskName
        self.funcMaskName = Parameters.funcMaskName
        self.funcDir = Parameters.funcDir
        self.funcRelPath = Parameters.funcRelPath
        self.funcName = Parameters.funcName
        self.subjectList = Parameters.subjectList
        self.outputPath = Parameters.outputPath
        self.useFuncMask = Parameters.useFuncMask
        self.nodeMode = Parameters.nodeMode
        self.cv = Parameters.cv
        self.fs = Parameters.fs
        self.kernel = Parameters.kernel
        self.estParameters = Parameters.estParameters
        self.givenC = Parameters.givenC
        self.givenE = Parameters.givenE
        self.conditionName = Parameters.conditionName
        self.numberCores = Parameters.numberCores

        # initiate the Subject dictionary that holds the name, age and path
        # of subjects used in this Study
        self.subjectParameters = {}
        self.subjects = {}
        # and of the networks
        self.networks = {}
        self.networkNames = None
        # and some to build the data structure
        self.subjectInfo = {}

        # lastly a path to itself after this thing has been saved so we know
        # how to get it back - get's filled upon saving
        self.path = None

    def getPaths(self):
        # a method to get the paths to the individual subjects files that are
        # to be used in the Preproc
        #
        # the paths are then stored
        listFile = open(self.subjectList)

        # loop through the Subject List line by line
        for line in listFile:
            # get the columns out, using default delimiter, and remove line
            # ending
            tempLine = line.strip().split()

            # assume that the first column contains the subjects name
            if len(tempLine) < 2:
                print ('!You did not give an age for some or all of your '
                       'subjects, SVR won\'t be available!')
                # need some breakpoint here if there is really only one colum
            tempSubject = tempLine[0]
            tempAge = np.fromstring(tempLine[1], dtype='float32', sep=' ')

            # other lines are currently not used for anything

            # generate path from Subject name an path information and store
            # it in a dictionary
            tempSubPath = os.path.join(self.funcDir, self.funcRelPath,
                                       tempSubject, self.funcName)
            self.subjectInfo[str(tempSubject)] = (tempSubject,
                                                  tempSubPath,
                                                  tempAge)

        # now do the same thing for the mask
        self.maskPath = os.path.join(self.maskDir, self.maskName)
        self.funcMaskPath = os.path.join(self.maskDir, self.funcMaskName)

    def makeMask(self):
        # method to load and separate into networks where applicable
        #
        # first, get load the mask file
        print 'Loading the file containing the mask '
        print 'I am using this path: \n', self.maskPath
        self.maskFile = nib.load(self.maskPath)
        tempMaskFile = self.maskFile.get_data()
        # by default this would be float32 and take huge amounts of memory
        self.maskData = np.asarray(tempMaskFile, dtype='int16')

        # see if we need the functional mask and get it accordingly
        if self.useFuncMask == 1:
            self.funcMaskFile = nib.load(self.funcMaskPath)
            tempFuncMask = self.funcMaskFile.get_data()
            # by default this would be float32 and take huge amounts of memory
            self.funcMaskData = np.asarray(tempFuncMask, dtype='int8')

        else:
            self.funcMaskData = None

        # If the mask is not a single 3D file, we need to identify and load
        # the Network masks
        if len(self.maskData.shape) == 4:
            # tell the user I detected a 4D file
            print '\n## 4D mask detected ##'
            print 'You have supplied a 4D mask file'
            print 'Masks are assumed to be stacked along the 4th dimension'
            print 'The first mask is assumed to be the whole brain mask and'
            print 'the following masks are assumed to be functional networks'
            print '## 4D mask detected ## \n'

            # Loop over 4th dimension and get all
            for networkNo in range(self.maskData.shape[3]):
                # if it's the first mask, we call it whole brain mask
                # all others get numerical values until I find a way to name
                # them dynamically
                # if networkNo == 0: - temporarily disabled
                if False:
                    # tempNet = self.maskData[..., networkNo]
                    # self.networks['whole brain'] = tempNet
                    pass
                else:
                    tempNet = self.maskData[..., networkNo]
                    self.networks[('Network ' + str(networkNo))] = tempNet

        elif len(self.maskData.shape) == 3:
            # tell the user I detected a 3D file
            print '\n## 3D mask detected ## \n'
            print 'You have supplied a 3D mask file '
            print 'I am assuming that this is a whole brain mask '
            print 'However, if it isn\'t, I don\'t care. But there will be '
            print 'only this one mask run. '
            print '\nStack your masks along the 4th dimension if you want to'
            print 'run multiple masks. '
            print '\n## 3D mask detected ## \n'

            self.networks['Whole brain Network'] = self.maskData

        else:
            # tell the user I can't do anything with the mask
            # file he supplied
            raise MaskError('Sorry, your mask file seems strange. I don\'t '
                            'know what to do with it ')

        # finally get the names of the networks that were just created
        self.networkNames = self.networks.keys()

    def giveMask(self):
        # method to force the mask dictionary down to the subjects
        for subject in self.subjects.values():
            subject.networksMasks = self.networks

    def makeSubjectParameters(self):
        # method to create and define the subject parameter objects
        #
        # first get the correct parameter object generated
        for subject in self.subjectInfo.keys():
            # read out the information for the current subject
            (tempSubject, tempSubPath, tempAge) = self.subjectInfo[subject]
            # instantiate a temporary parameter object
            tempParam = Parameters(str(tempSubject))
            # and put the variables in
            tempParam.makeSubject(subjectPath=tempSubPath,
                                  age=tempAge,
                                  networks=self.networks,
                                  useFuncMask=self.useFuncMask,
                                  funcMaskData=self.funcMaskData,
                                  nodeMode=self.nodeMode)

            # store them in the parameter dictionary
            self.subjectParameters[subject] = tempParam

    def makeSubject(self):
        # method to generate subject objects from the list of parameters
        for subject in self.subjectParameters.keys():
            # get the parameter object from the dictionary
            tempParam = self.subjectParameters[subject]
            # create a subject instance, put it in a temporary object and
            # then into the subject dictionary
            tempObj = Subject(tempParam)
            self.subjects[subject] = tempObj

            # clean up the parameter object as we won't need it again
            self.subjectParameters[subject] = None

    def runSubject(self, subject):
        # method to execute the internal methods of the subject objects
        # this is used for parallel processing

        subject.getTimeseries()

        subject.getConnectivity()

        subject.getFeatures()

    def runPreprocessing(self):
        # method to execute the parallel preprocessing on the subjects
        #
        # build the argument list
        argumentList = []

        for subject in self.subjects.values():
            argumentList.append(subject)

        debug = 0
        if debug == 1:
            for subject in self.subjects.keys():
                tempSubj = self.subjects.get(subject)
                print 'currently subject', subject
                print type(tempSubj)
                self.runSubject(tempSubj)
            print 'Done with that'
        else:
            print 'Starting multicore processing with', str(self.numberCores)
            start = time.time()
            pool = mp.Pool(processes=self.numberCores)
            resultList = pool.map(self.runSubject, argumentList)
            stop = time.time()
            elapsed = stop - start
            print 'Multicore Processing done. Took', str(elapsed), 'seconds'
            print len(resultList)

    def saveYourself(self):
        # a method to dump out the whole object onto disk and then pass the
        # path into the directory
        pass

    def cleanup(self):
        # method for cleaning up the elements of the analysis, this should at
        # some point have different levels of cleanup
        #
        # clean up no longer needed information inside the subjects
        for subName in self.subjects.keys():
            # get the subject
            subject = self.subjects[subName]
            # remove the connectivity matrix
            subject.connectivity = None
            # remove the timeseries of all nodes
            subject.networStack = None
            # loop over the networks inside the subject
            for netName in subject.networks.keys():
                # get the network
                network = subject.networks[netName]
                # remove network timeseries
                network.timeseries = None
                # remove connectivity
                network.connectivity = None
                # remove mask data
                network.maskData = None
                # remove the features I don't want
                for feature in network.features.keys():
                    if feature == 'whole' or feature == 'order whole':
                        pass
                    else:
                        # get rid of the features
                        network.features[feature] = None

            # remove the mask data in the subject
            self.maskData = None
            self.funcMaskData = None
            self.networksMasks = None
            # and lastly get rid of the indices
            self.networkIndices = None

        # cleanup on the analysis level
        self.networks = None
        self.subjectParameters = None
        # just for testing, get rid of everything below the analysis level
        # self.subjects = None

    def purge(self):
        # seriously get rid of everything
        self.maskDir = None
        self.maskName = None
        self.funcMaskName = None
        self.funcDir = None
        self.funcRelPath = None
        self.funcName = None
        self.subjectList = None
        self.outputPath = None
        self.useFuncMask = None
        self.nodeMode = None
        self.estParameters = None
        self.givenC = None
        self.givenE = None
        self.conditionName = None
        self.numberCores = None

        self.subjectParameters = None
        self.subjects = None
        # and of the networks
        self.networks = None
        self.networkNames = None
        # and some to build the data structure
        self.subjectInfo = None


class Subject(object):
    # class for Subject objects. These objects store the Subject information,
    # functional data and - after preprocessing - the connectivity data
    # for the networks used in the Preproc
    def __init__(self, Parameters):
        self.name = Parameters.name
        self.path = Parameters.subjectPath
        self.age = Parameters.age
        self.networksMasks = Parameters.networks
        self.useFuncMask = Parameters.useFuncMask
        self.funcMaskData = Parameters.funcMaskData
        self.nodeMode = Parameters.nodeMode

        # prepare the networks dictionary
        self.networks = {}

        # fill the Network dictionary with the Storage object
        for network in self.networksMasks.keys():
            networkMask = self.networksMasks[network]
            networkStorage = Storage(network)
            networkStorage.putNetwork(networkMask)
            self.networks[network] = networkStorage

    def getTimeseries(self):
        # method to extract the timeseries from the functional data according
        # to the masks

        # print '    getting timeseries', self.name

        self.file = nib.load(self.path)
        self.data = self.file.get_data()

        # now we loop over the networks dictionary and get the timeseries
        # for each of their nodes
        for network in self.networks.keys():

            # inside this loop we keep everything in temporary variables
            # e.g. nothing gets assigned as attribute to the object itself
            #
            # prepare the timeseries array
            timeseries = np.array([])
            tempNet = self.networks[network]

            # check if we need to use the functional mask
            if self.useFuncMask == 1:
                tempNet.maskData = tempNet.maskData * self.funcMaskData
            else:
                # if not, just set all zero-voxels in the first timepoint to
                # zero in the mask data
                tempNet.maskData[self.data[..., 0] == 0] = 0

            # create a vector of unique elements in the mask that are not 0
            nodeVector = np.unique(tempNet.maskData[tempNet.maskData != 0])

            # now we loop over the nodes in the mask
            for node in nodeVector:
                # first get an array of all the voxels in the current element
                # times the number of timepoints in the timeseries
                nodeIndex = np.where(tempNet.maskData == node)
                # get the functional data corresponding to the current node
                tempData = self.data[nodeIndex]
                # then average the timeseries
                tempAverage = np.mean(tempData, axis=0).astype('float32')
                if timeseries.size == 0:
                    timeseries = tempAverage[np.newaxis, ...]
                else:
                    timeseries = np.concatenate((timeseries,
                                               tempAverage[np.newaxis, ...]),
                                               axis=0)

            # now we have the timeseries of dimension nodes by timepoints
            # this can get stored in the Network Storage object
            self.networks[network].timeseries = timeseries
            self.networks[network].nodeVector = nodeVector
        # get rid of the functional data again
        self.file = None
        self.data = None

    def getConnectivity(self):
        # method to get the connectivity after extracting the timeseries
        #
        # we loop over the Network Storage objects in the Subject and stack
        # them together

        # print '    getting Connectivity', self.name

        networkKeys = self.networks.keys()
        # we only want the networks, not the whole brain - temporarily disabled
        # networkList = [key for key in networkKeys if 'Network' in key]
        networkList = networkKeys
        # prepare an array to stack the timeseries together to compute the
        # full connectivity matrix
        self.networkStack = np.array([])
        # prepare the dictionary for the Network indices to get them back
        # from the correlation matrix later
        self.networkIndices = {}

        for network in networkList:
            # get the Network object ...
            tempNet = self.networks[network]
            # ... and stack its timeseries
            tempTimeseries = tempNet.timeseries
            # get the current shape of the stack for the index
            stackShape = self.networkStack.shape[0]
            # get the shape of the temporary timeseries
            timeShape = tempTimeseries.shape[0]
            # store the index of the current Network in the stack
            networkIndex = (stackShape, stackShape + timeShape)
            # store both in the Subject object and the Network Storage object
            tempNet.index = networkIndex
            self.networkIndices[network] = networkIndex
            # append the current timeseries to the stack
            if self.networkStack.size == 0:
                # this is the first instance, so we don't append anything but
                # just set the stack to the first timeseries
                self.networkStack = tempTimeseries
            else:
                self.networkStack = np.concatenate((self.networkStack,
                                                    tempTimeseries),
                                                   axis=0)

        # make the full connectivity matrix
        tempConnectivity = np.corrcoef(self.networkStack)
        # z-transform the connections with fisher's z-transformation
        self.connectivity = np.arctanh(tempConnectivity)
        # try getting the size down
        tempCon = self.connectivity
        self.connectivity = np.asarray(tempCon, dtype='float16')

        # dump the connectivity
        #dumpFile = gzip.open((self.name + 'connect'), 'wb')
        #cPickle.dump(self.connectivity, dumpFile)

        # get rid of the networkStack
        self.networkStack = None

        # we will loop over the networks again to extract the correct edges
        for target in networkList:
            # get the indices for the target
            ind = self.networkIndices[target]
            targetRow = self.connectivity[ind[0]:ind[1], ]  # possible error:
            #
            # get a local copy of the Network object
            tempNet = self.networks[target]
            # declare the target Network in the connectivity dictionary
            tempNet.connectivity['target'] = ('target Network is', target)

            for network in networkList:
                # get the Network index
                tempInd = self.networkIndices[network]
                # get the correct section of the Network rows
                networkMatrix = targetRow[:, tempInd[0]:tempInd[1]]
                if network == target:
                    # because this is connectivity within, we only want the
                    # lower triangle of the matrix for the feature vector
                    # first get an index matrix of the same dimensions
                    index = np.ones(networkMatrix.shape, dtype='int')
                    # strip the index matrix from the upper triangle and diag
                    index = np.tril(index, -1)
                    # now get the target vector from all the locations that
                    # are not zero
                    # note: this goes row-wise!
                    tempVector = networkMatrix[index != 0]
                    networkVector = np.asarray(tempVector, dtype='float16')
                    # store it in the connectivity dictionary of the Network
                    # Storage object
                    tempNet.connectivity[network] = networkVector
                else:
                    # if we are not handling the target Network, we do not
                    # need to get only the lower triangle but can take all
                    tempVector = networkMatrix.flatten()
                    networkVector = np.asarray(tempVector, dtype='float16')
                    # store it in the connectivity dictionary of the Network
                    # Storage object

                    tempNet.connectivity[network] = networkVector

        # dump networks
        #dumpFile = gzip.open((self.name + 'networks'), 'wb')
        #cPickle.dump(self.networks, dumpFile)

    def getFeatures(self):
        # method to generate feature vectors for between-, within-, and
        # whole-Network connectivity. Later, this could become more dynamic
        # and allow for a selection of networks to be included in the feature
        #
        # we loop over the Network Storage objects in the Subject

        # print '    getting Features', self.name

        networkKeys = self.networks.keys()
        # we only want the networks, not the whole brain - temporarily disabled
        # networkList = [key for key in networkKeys if 'Network' in key]
        networkList = networkKeys

        for network in networkList:
            # load the Network Storage object and keep it as a temp file
            tempNet = self.networks[network]
            # prepare the Storage for the different keys that get appended to
            tempNet.features['whole'] = np.array([])
            tempNet.features['order whole'] = []
            tempNet.features['between'] = np.array([])
            tempNet.features['order between'] = []

            # secondary loop for the files inside the file
            for storeNet in networkList:
                # check what kind of connectivity this is and then store
                # accordingly
                tempConnectivity = tempNet.connectivity[storeNet]
                # regardless of type, all connectivity goes into 'whole'
                tempNet.features['whole'] = np.append(
                                            tempNet.features['whole'],
                                            tempConnectivity)
                # also store the name of the Network to have the order
                tempNet.features['order whole'].append(storeNet)
                if storeNet == network:
                    # the current Network is within connectivity
                    tempNet.features['within'] = tempConnectivity
                else:
                    # the current Network is between connectivity
                    tempNet.features['between'] = np.append(
                                                tempNet.features['between'],
                                                tempConnectivity)
                    # also store the name of the Network to have the order
                    tempNet.features['order between'].append(storeNet)

            # map the stuff back
            self.networks[network] = tempNet


class Analysis(object):
    # class to contain and control the analysis of the preprocessed data
    def __init__(self, name):

        self.name = name
        # storage objects for data inside this dictionary that contains all
        # the networks that will be run
        self.networks = {}

        # lastly put the path to the dumped file - get's passed when dumped
        self.path = None

        # and some running parameters initialized
        self.nodeMode = None
        self.cv = None
        self.fs = None
        self.estParameters = None
        self.kernel = None

    def run(self):
        # method that runs whatever machines are inside the networks dictionary
        for networkName in self.networks.keys():
            network = self.networks[networkName]
            # run stuff
            network.makeFolds()
            network.runFolds()

    def cleanup(self):
        # clean up the object
        pass

    def purge(self):
        # seriously get rid of everything
        self.networks = None
        pass


class Network(object):
    # a class for the machine learning part of the analysis
    # this just gets build and then we see if it makes sense
    def __init__(self,
                 name,
                 subjects,
                 cv,
                 fs,
                 kernel,
                 estParameters,
                 givenC,
                 givenE,
                 numberCores):
        self.name = name
        self.subjects = subjects
        self.cv = cv
        self.fs = fs
        self.kernel = kernel
        self.estParameters = estParameters
        self.numberCores = numberCores
        self.cvObject = None
        self.folds = {}
        self.numberFolds = 10
        self.predictedData = np.array([])
        self.trueData = np.array([])

        # tempStore for the given parameters
        self.C = givenC
        self.E = givenE

        # create a name array of values that we can loop over
        self.subNames = self.subjects.keys()

    def makeFolds(self):
        # method to create the folds for the cross-validation
        # uses the subject object from the analysis level as a temporary source
        #
        # monitor memory here, if it gets blown up, clean quicker

        # make a cv object - later this could be done in a separate method
        if self.cv == 1:
            print 'Using LOOCV for crossvalidation'
            self.cvObject = LeaveOneOut(len(self.subNames))
        else:
            print 'Using Kfold for crossvalidation'
            # if not, we just use the normal k-fold
            if len(self.subNames) < self.numberFolds:
                self.numberFolds = len(self.subNames)

            self.cvObject = KFold(len(self.subNames),
                                  self.numberFolds,
                                  shuffle=True)

        run = 1
        for cvInstance in self.cvObject:
            # create a new instance of the fold class
            fold = Fold(self.fs)
            fold.kernel = self.kernel
            train = cvInstance[0]
            test = cvInstance[1]
            # now we loop over these
            for subId in train:

                # append this subject to the train arrays
                # can be extended to include more than one network possibly
                subName = self.subNames[subId]
                subject = self.subjects[subName]
                feature = subject.feature
                age = subject.age
                # store the feature in the feature array of the fold
                if fold.trainFeatures.size == 0:
                    # this is the first instance, so we don't append anything
                    # but just set the stack to the first timeseries
                    fold.trainFeatures = feature[np.newaxis, ...]
                else:
                    fold.trainFeatures = np.concatenate((fold.trainFeatures,
                                                    feature[np.newaxis, ...]),
                                                    axis=0)
                # also store the age
                fold.trainAges = np.append(fold.trainAges, age)

            for subId in test:
                # append this subject to the test arrays
                # append this subject to the train arrays
                # can be extended to include more than one network possibly
                subName = self.subNames[subId]
                subject = self.subjects[subName]
                feature = subject.feature
                age = subject.age
                # store the feature in the feature array of the fold
                if fold.testFeatures.size == 0:
                    # this is the first instance, so we don't append anything
                    # but just set the stack to the first timeseries
                    fold.testFeatures = feature[np.newaxis, ...]
                else:
                    fold.testFeatures = np.concatenate((fold.testFeatures,
                                                    feature[np.newaxis, ...]),
                                                    axis=0)
                # also store the age
                fold.testAges = np.append(fold.testAges, age)

            # lastly store the kfold object
            fold.kfold = cvInstance
            fold.cv = self.cv
            fold.estParameters = self.estParameters
            fold.C = self.C
            fold.E = self.E
            fold.numberCores = self.numberCores
            # store the fold in the dictionary
            foldName = str(run)
            self.folds[foldName] = fold
            run += 1

        # print that this has finished
        print 'Making folds completed'

    def runFolds(self):
        # method to actually run the folds
        for foldName in sorted(self.folds.iterkeys()):
            fold = self.folds[foldName]
            # print '\n#### Running', foldName, '####'

            # start running stuff
            fold.selectFeatures()
            # depending on whether or not we actually estimate the parameters
            if self.estParameters == 1:
                fold.estimateParameters()
            else:
                fold.C = self.C
                fold.E = self.E
            print 'Training model'
            fold.trainModel()
            print 'Testing model'
            fold.testModel()
            # getting the predicted data together again
            print '#### Done running', foldName, self.name, '####\n'
            self.predictedData = np.append(self.predictedData,
                                           fold.predictedAges)
            self.trueData = np.append(self.trueData, fold.testAges)
            fold.cleanup()

        # print the number of predicted and true ages
        #print ('\n##### Done running '
        #       + self.name
        #       + ' #####')
        #print ('The number of true ages is: '
        #       + str(self.trueData.shape[0])
        #       + '\nand the number of predicted ages is: '
        #       + str(self.predictedData.shape[0]))


class Fold(object):
    # class to keep the fold objects for the iterations
    # fuck it, I'll just call it fold for now
    def __init__(self, fs):
        # initialize, mostly preparation of the attributes without assigning
        # anything

        # first the parameters for training
        self.trainSubjects = []
        self.trainFeatures = np.array([])
        self.trainAges = np.array([])
        # and then for testing
        self.testSubjects = None
        self.testFeatures = np.array([])
        self.testAges = np.array([])
        # and lastly for the results
        self.predictedAges = np.array([])
        self.featureIndex = np.array([])
        self.featureWeights = np.array([])
        self.model = None

        # parameter attributes for the algorithm
        self.kernel = None
        self.C = None
        self.E = None

        # and a copy of the subject index
        self.kfold = None

        # and some other parameters
        self.numberCores = 2
        self.fs = fs

    def cleanup(self):
        # clean up all the temporary objects that we don't need after one run
        # first the parameters for training
        self.trainSubjects = []
        self.trainFeatures = None
        self.trainAges = None
        # and for testing
        self.testSubjects = None
        self.testFeatures = None
        self.testAges = None

    def selectFeatures(self):
        # method to run on the data of each established fold in order to get
        # the most predictive features for the subjects in the training set
        #
        # get the number of original features
        featureNumber = self.trainFeatures.shape[1]

        if self.fs == 0 or self.kernel != 'linear':
            if self.kernel != 'linear' and self.fs == 1:
                # hit a warning because we can't do this
                print '\n\n##### WARNING #####'
                print 'dude, your feature selection settings don\'t go well '
                print 'with your selected kernel'
                print 'You want to run RFE but chose a non-linear kernel.'
                print 'won\'t work! seriously! Reconsider'
                print '##### WARNING #####'
                time.sleep(3)

            else:
                # apparently we really intended to do this
                print '\n\n##### Note #####'
                print 'dude, you really want to go there... '
                print 'You can run this correlation based feature selection '
                print 'but what would Jesus do?'
                print 'It\'s not a nice way of doing things. Have a nice day! '
                print '##### Note #####'

            # we have to loop over the entire feature vector
            # first we need a storage variable
            storage = np.array([])
            for run in range(featureNumber):
                corr = st.pearsonr(self.trainFeatures[:, run],
                                   self.trainAges)[0]

                storage = np.append(storage, corr)
            # so now we have all the correlation values of each feature with
            # age we are only interested in the magnitude so we need
            # absolute values
            absStore = np.absolute(storage)

            # and of this I want the top 200. so first we do an argsort
            # and take the last/highest 200 values
            absIndex = np.argsort(absStore)[-200:]

            # now get these values from the feature vector and sell
            # the index as the feature index
            tempIndex = np.zeros(absStore.shape, dtype=int)
            tempIndex[absIndex] = 1
            featureIndex = np.copy(tempIndex)
            pass
        # if we are using the recursive feature elimination, do this
        # (also check for kernel)
        elif self.fs == 1 and self.kernel == 'linear':

            # instantiate the estimator for the feature selection
            # fixed kernel has to go in the future
            svrEstimator = svm.SVR(kernel=self.kernel)
            # if this number is to big, reduce it quickly without cv
            if featureNumber >= 1000:
                # shrink it down
                rfeObject = RFE(estimator=svrEstimator,
                                n_features_to_select=1000,
                                step=0.1)

                rfeObject.fit(self.trainFeatures, self.trainAges)
                tempRfeIndex = rfeObject.support_
                rfeIndex = np.where(tempRfeIndex)[0]
                # reduce a temporary copy of the features to the selection
                tempTrainFeatures = self.trainFeatures[..., rfeIndex]

                # now run the crossvalidated feature selection on the data
                rfecvObject = RFECV(estimator=svrEstimator,
                                    step=0.01,
                                    cv=2,
                                    loss_func=mean_squared_error)
                rfecvObject.fit(tempTrainFeatures, self.trainAges)
                tempRfeCvIndex = rfecvObject.support_
                rfeCvIndex = np.where(tempRfeCvIndex)[0]
                # apply the index only to the original features so it can be
                # mapped back better and also used for the testing set
                #
                # create an integer index

                tempIndex = np.zeros(featureNumber, dtype=int)
                tempIndex[rfeIndex[rfeCvIndex]] = 1
                featureIndex = tempIndex

                #
                # Note! remapping is not yet possible with this code
                #

            else:
                # no need for the initial shrinking, otherwise the same
                rfecvObject = RFECV(estimator=svrEstimator,
                                    step=0.01,
                                    cv=2,
                                    loss_func=mean_squared_error)
                rfecvObject.fit(self.trainFeatures, self.trainAges)
                tempRfeCvIndex = rfecvObject.support_
                rfeCvIndex = np.where(tempRfeCvIndex)[0]

                tempIndex = np.zeros(featureNumber, dtype=int)
                tempIndex[rfeCvIndex] = 1
                featureIndex = tempIndex

        # assign the values back to the object
        bestTrainFeatures = self.trainFeatures[..., featureIndex == 1]
        self.trainFeatures = bestTrainFeatures
        self.featureIndex = featureIndex
        # and do the same to the test features
        bestTestFeatures = self.testFeatures[..., self.featureIndex == 1]
        self.testFeatures = bestTestFeatures
        #print 'new features:', self.trainFeatures.shape[1]
        #print ('Fold uses '
        #       + str(self.trainFeatures.shape[0])
        #       + ' subjects for training'
        #       + '\nand '
        #       + str(self.testFeatures.shape[0])
        #       + ' subjects for testing')

    def estimateParameters(self):
        # method to estimate the optimal parameters for the algorithm
        #
        # currently hardwired number of crossvalidations
        cv = 5
        if len(self.trainAges) < cv:
            cv = len(self.trainAges)

        # provide the parameters for the first, coarse pass
        # set of parameters
        expArrayOne = np.arange(-4, 4, 1)
        baseArray = np.ones_like(expArrayOne, dtype='float32') * 10
        parameterOne = np.power(baseArray, expArrayOne).tolist()

        # if for some reason this doesn't work, just paste directly
        parameters = {'C': parameterOne}
        gridModel = svm.SVR(kernel=self.kernel, epsilon=self.E)

        #print '\nFirst pass parameter estimation! '
        firstTrainModel = grid_search.GridSearchCV(gridModel,
                                                   parameters,
                                                   cv=cv,
                                                   n_jobs=self.numberCores,
                                                   verbose=0)
        firstTrainModel.fit(self.trainFeatures, self.trainAges)
        firstPassC = firstTrainModel.best_estimator_.C

        # print 'The Firstpass C parameter is:', firstPassC

        expFirstC = np.log10(firstPassC)
        expArrayTwo = np.arange(expFirstC - 1, expFirstC + 1.1, 0.1)
        baseArrayTwo = np.ones_like(expArrayTwo, dtype='float32') * 10

        parameterTwo = np.power(baseArrayTwo, expArrayTwo).tolist()

        # in case this causes trouble, paste directly
        parameters = {'C': parameterTwo}

        # reuse estimated parameters for second, better pass
        #print 'Second pass parameter estimation! '

        secondTrainModel = grid_search.GridSearchCV(gridModel,
                                                    parameters,
                                                    cv=cv,
                                                    n_jobs=self.numberCores,
                                                    verbose=0)
        secondTrainModel.fit(self.trainFeatures, self.trainAges)
        bestC = secondTrainModel.best_estimator_.C
        #print 'Overall best C parameter is:', bestC
        # write it into the object
        self.C = bestC

    def trainModel(self):
        # method to actually train the model with the extracted features and
        # the estimated parameters
        trainModel = svm.SVR(kernel=self.kernel, C=self.C, epsilon=self.E)
        trainModel.fit(self.trainFeatures, self.trainAges)
        self.model = trainModel

    def testModel(self):
        # method to predict the ages from the testing data
        self.predictedAges = self.model.predict(self.testFeatures)
