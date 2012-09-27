# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:58:26 2012

@author: sebastian

classes for performing the network based maturity analysis

"""
### Imports ###
import os
import re
import time
import numpy as np
import nibabel as nib
from multiprocessing import Pool


### Error handling ###
class MaskError(Exception):
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return repr(self.text)


### Service Objects ###
class parameters(object):

    def __init__(self, name):
        # here, we just create the object and give it a name
        self.name = name

    def makeStudy(self,
                  configPath):

        self.configPath = configPath

    def makeAnalysis(self,
                     maskPath,
                     maskName,
                     funcMaskName,
                     funcPath,
                     funcRelPath,
                     funcName,
                     subjectList,
                     outputPath,
                     useFuncMask,
                     nodeMode,
                     estParameters,
                     givenC,
                     givenE,
                     conditionName):
        # define all the parameters
        self.maskPath = maskPath
        self.maskName = maskName
        self.funcMaskName = funcMaskName
        self.funcPath = funcPath
        self.funcRelPath = funcRelPath
        self.funcName = funcName
        self.subjectList = subjectList
        self.outputPath = outputPath
        self.useFuncMask = useFuncMask
        self.nodeMode = nodeMode
        self.estParameters = estParameters
        self.givenC = givenC
        self.givenE = givenE
        self.conditionName = conditionName

    def makeSubject(self,
                    subjectPath,
                    age,
                    networks,
                    useFuncMask,
                    funcMaskData,
                    nodeMode):
        # method to generate subject parameters
        self.subjectPath = subjectPath
        self.age = age
        self.networks = networks
        self.useFuncMask = useFuncMask
        self.funcMaskData = funcMaskData
        self.nodeMode = nodeMode

    def makeNetwork(self):
        pass

    def introduce(self):
        # method for reporting the current parameters of the object
        print 'Hi, my name is', self.name
        print 'my parameters are:', self.__dict__


class storage(object):
    # a class to store information that is too complex for dictionaries
    def __init__(self, name):
        self.name = name

    def putNetwork(self,
                   maskData):

        self.type = 'network'
        self.maskData = maskData

        # initiating attributes that can be filled later
        self.timeseries = None
        self.nodeVector = None
        self.index = None
        self.networkVector = None
        self.connectivity = {}
        self.features = {}

    def putSubject(self,
                   age):

        self.type = 'subject'
        self.age = age

### Data objects ###


class study(object):
    # class for the study object that contains all the analyses and the
    # list of parameters to control these analyses
    def __init__(self, parameters):

        self.name = parameters.name
        self.configPath = parameters.configPath
        self.numberCores = parameters.numberCores

        # initialize the parameter dictionary for the analyses
        self.analysisParameters = {}
        self.analyses = {}

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
            if '#' in line or re.match('^\n', line):
                print 'Not using this:', line.strip()
            # check if right number of commands are parsed
            elif len(line.strip().split()) != 14:
                print '\nWrong number of commands '
            else:
                print '\nCommand OK! '
                commandString = line.strip().split()
                maskPath = commandString[0]
                maskName = commandString[1]
                funcMaskName = commandString[2]
                funcPath = commandString[3]
                funcRelPath = commandString[4]
                funcName = commandString[5]
                subjectList = commandString[6]
                outputPath = commandString[7]
                useFuncMask = int(commandString[8])
                nodeMode = int(commandString[9])
                estParameters = int(commandString[10])
                givenC = float(commandString[11])
                givenE = float(commandString[12])
                conditionName = commandString[13]

                # initialize the parameter object inside a temporary variable
                tempObj = parameters(conditionName)
                # and fill in the parameters that were just read in
                tempObj.makeAnalysis(maskPath=maskPath,
                                     maskName=maskName,
                                     funcMaskName=funcMaskName,
                                     funcPath=funcPath,
                                     funcRelPath=funcRelPath,
                                     funcName=funcName,
                                     subjectList=subjectList,
                                     outputPath=outputPath,
                                     useFuncMask=useFuncMask,
                                     nodeMode=nodeMode,
                                     estParameters=estParameters,
                                     givenC=givenC,
                                     givenE=givenE,
                                     conditionName=conditionName)
                # store the complete parameter object in the dictionary
                self.analysisParameters[conditionName] = tempObj

    def makeAnalysis(self):
        # method to loop over the dictionary with parameter objects and
        # create the corresponding analysis objects
        #
        # the process is rather sipmle, just create a new instance of the
        # analysis class and hand over the parameter object
        for analysis in self.analysisParameters.keys():
            # get the respective parameter object from the dictionary
            tempParameter = self.analysisParameters[analysis]
            # put the analysis object in a temporary object
            tempObj = analysis(tempParameter)
            # and save it to the analyses dictionary
            self.analyses[analysis] = tempObj


class analysis(object):
    # class for analysis objects that contain network and subject objects
    # and are themselves stored in the study object
    def __init__(self, parameters):
        self.name = parameters.name
        self.maskPath = parameters.maskPath
        self.maskName = parameters.maskName
        self.funcMaskName = parameters.funcMaskName
        self.funcPath = parameters.funcMask
        self.funcRelPath = parameters.funcRelPath
        self.funcName = parameters.funcName
        self.subjectList = parameters.subjectList
        self.outputPath = parameters.outputPath
        self.useFuncMask = parameters.useFuncMask
        self.nodeMode = parameters.nodeMode
        self.estParameters = parameters.estParameters
        self.givenC = parameters.givenC
        self.givenE = parameters.givenE
        self.conditionName = parameters.conditionName
        self.numberCores = parameters.numberCores

        # initiate the subject dictionary that holds the name, age and path
        # of subjects used in this study
        self.subjects = {}
        # and of the networks
        self.networks = {}
        # and some to build the data structure
        self.subjectInfo = {}

    def getPaths(self):
        # a method to get the paths to the individual subjects files that are
        # to be used in the analysis
        #
        # the paths are then stored
        listFile = open(self.subjectList)

        # loop through the subject List line by line
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

            # generate path from subject name an path information and store
            # it in a dictionary
            tempSubPath = os.path.join(self.funcDir, self.funcRelPath,
                                       self.funcName)
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
        self.maskData = self.maskFile.get_data()

        # see if we need the functional mask and get it accordingly
        if self.useFuncMask == 1:
            self.funcMaskFile = nib.load(self.funcMaskPath)
            self.funcMaskData = self.funcMaskFile.get_data()

        else:
            self.funcMaskData = None

        # If the mask is not a single 3D file, we need to identify and load
        # the network masks
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
                if networkNo == 0:
                    self.networks['whole brain'] = self.maskData[...,
                                                                 networkNo]
                else:
                    self.networks[('network ',
                                  str(networkNo))] = self.maskData[...,
                                                                 networkNo]

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

            self.networks['whole brain'] = self.maskData

        else:
            # tell the user I can't do anything with the mask
            # file he supplied
            raise MaskError('Sorry, your mask file seems strange. I don\'t '
                            'know what to do with it ')

    def makeSubject(self):
        # method to create and define the subject objects
        #
        # first get the correct parameter object generated
        for subject in self.subjectInfo.keys():
            # read out the information for the current subject
            (tempSubject, tempSubPath, tempAge) = self.subjectInfo[subject]
            # instantiate a temporary parameter object
            tempParam = parameters(str(tempSubject))
            # and put the variables in
            tempParam.makeSubject(subjectPath=tempSubPath,
                                  age=tempAge,
                                  networks=self.networks,
                                  useFuncMask=self.useFuncMask,
                                  funcMaskData=self.funcMaskData,
                                  nodeMode=self.nodeMode)

            # put them in a temporary object and then into
            # the subject dictionary
            tempObj = subject(tempParam)
            self.subjects[subject] = tempObj

    def runSubject(self, subject):
        # method to execute the internal methods of the subject objects
        # this is used for parallel processing
        subject.getTimeseries()
        subject.getConnectivity()
        subject.getFeatures
        # this should take care of it all

    def runPreprocessing(self):
        # method to execute the parallel preprocessing on the subjects
        #
        # build the argument list
        argumentList = []

        for subject in self.subjects.values():
            argumentList.append(subject)

        print 'Starting multicore processing with', str(self.numberCores)
        start = time.time()
        pool = Pool(processes=self.numberCores)
        resultList = pool.map(self.runSubject, argumentList)
        stop = time.time()
        elapsed = stop - start
        print 'Multicore Processing done. Took', str(elapsed), 'seconds'
        print len(resultList)


class subject(object):
    # class for subject objects. These objects store the subject information,
    # functional data and - after preprocessing - the connectivity data
    # for the networks used in the analysis
    def __init__(self, parameters):
        self.name = parameters.name
        self.path = parameters.subjectPath
        self.age = parameters.age
        self.networksMasks = parameters.networks
        self.useFuncMask = parameters.useFuncMask
        self.funcMaskData = parameters.funcMaskData
        self.nodeMode = parameters.nodeMode

        # prepare the networks dictionary
        self.networks = {}

        # fill the network dictionary with the storage object
        for network in self.networksMasks.keys():
            networkMask = self.networksMasks[network]
            networkStorage = storage(network)
            networkStorage.putNetwork(networkMask)
            self.networks[network] = networkStorage

    def getTimeseries(self):
        # method to extract the timeseries from the functional data according
        # to the masks
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
                tempNet.maskData = tempNet.masData * self.funcMaskData
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
            # this can get stored in the network storage object
            self.networks[network].timeseries = timeseries
            self.networks[network].nodeVector = nodeVector

    def getConnectivity(self):
        # method to get the connectivity after extracting the timeseries
        #
        # we loop over the network storage objects in the subject and stack
        # them together
        networkKeys = self.networks.keys()
        # we only want the networks, not the whole brain
        networkList = [key for key in networkKeys if 'network' in key]
        # prepare an array to stack the timeseries together to compute the
        # full connectivity matrix
        self.networkStack = np.array([])
        # prepare the dictionary for the network indices to get them back
        # from the correlation matrix later
        self.networkIndices = {}

        for network in networkList:
            # get the network object ...
            tempNet = self.networks[network]
            # ... and stack its timeseries
            tempTimeseries = tempNet.timeseries
            # get the current shape of the stack for the index
            stackShape = self.networkStack.shape[0]
            # get the shape of the temporary timeseries
            timeShape = tempTimeseries.shape[0]
            # store the index of the current network in the stack
            networkIndex = (stackShape, stackShape + timeShape)
            # store both in the subject object and the network storage object
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
        self.connectivity = np.corrcoef(self.networkStack)

        # we will loop over the networks again to extract the correct edges
        for target in networkList:
            # get the indices for the target
            ind = self.networkIndices[target]
            targetRow = self.connectivity[ind[0]:ind[1], ]  # possible error:
            #
            # get a local copy of the network object
            tempNet = self.networks[target]
            # declare the target network in the connectivity dictionary
            tempNet.connectivity['target'] = ('target network is', target)

            for network in networkList:
                # get the network index
                tempInd = self.networkIndices[network]
                # get the correct section of the network rows
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
                    networkVector = networkMatrix[index != 0]
                    # store it in the connectivity dictionary of the network
                    # storage object
                    tempNet.connectivity[network] = networkVector
                else:
                    # if we are not handling the target network, we do not
                    # need to get only the lower triangle but can take all
                    networkVector = networkMatrix.flatten()
                    # store it in the connectivity dictionary of the network
                    # storage object
                    tempNet.connectivity[network] = networkVector

    def getFeatures(self):
        # method to generate feature vectors for between-, within-, and
        # whole-network connectivity. Later, this could become more dynamic
        # and allow for a selection of networks to be included in the feature
        #
        # we loop over the network storage objects in the subject
        networkKeys = self.networks.keys()
        # we only want the networks, not the whole brain
        networkList = [key for key in networkKeys if 'network' in key]

        for network in networkList:
            # load the network storage object and keep it as a temp file
            tempNet = self.networks[network]
            # prepare the storage for the different keys
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
                # also store the name of the network to have the order
                tempNet.features['order whole'].append(storeNet)
                if storeNet == network:
                    # the current network is within connectivity
                    tempNet.features['within'] = tempConnectivity
                else:
                    # the current network is between connectivity
                    tempNet.features['between'] = np.append(
                                                tempNet.features['between'],
                                                tempConnectivity)
                    # also store the name of the network to have the order
                    tempNet.features['order between'].append(storeNet)


class network(object):
    # class for network objects. These objects each store on of the masks
    # used in the analysis and - after preprocessing - the connectivity
    # data generated by the subject-object together with demographic info.
    def __init__(self):
        pass