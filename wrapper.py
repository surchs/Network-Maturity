# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:14:19 2012

@author: sebastian

A wrapper to run all scripts
"""
# get all the file imports
import re
import sys
import time
import preprocesser_wrap as pp
import machinetamer_wrap as mt


def Wrapper(maskPath,
            maskName,
            funcPath,
            funcRelPath,
            funcName,
            subjectList,
            outputPath,
            nodeMode,
            estParameters,
            givenC,
            givenE,
            conditionName,
            numberCores,
            saveFiles=True):
    # introduce ourselves
    print '\n## '
    print 'Running Wrapper for condition:', conditionName
    print '##\n '

    # run the first step - preprocessing
    preprocDict = pp.Main(maskPath,
                          maskName,
                          funcPath,
                          funcRelPath,
                          funcName,
                          subjectList,
                          outputPath,
                          conditionName,
                          numberCores,
                          nodeMode)

    print 'waiting for 5 seconds to give the pools some time to cool down.'
    time.sleep(5)
    print 'done sleeping, let\'s roll '

    # run second step - machine learning
    if estParameters == 0:
        machineDict = mt.Main(preprocDict,
                              conditionName,
                              outputPath,
                              numberCores=numberCores,
                              estParameters=estParameters,
                              givenC=givenC,
                              givenE=givenE)

    else:
        machineDict = mt.Main(preprocDict,
                              conditionName,
                              outputPath,
                              numberCores=numberCores)

    return (preprocDict, machineDict)


def Main(configFile, numberCores=8):
    # introduction
    print '\nWelcome to the Wrapper '
    print 'today, we are running with', numberCores, 'cores for every process '
    print 'have a nice day \n '

    # save Files are set up globally
    saveFiles = True
    print 'Saving Files? =', saveFiles

    # load the configfile
    config = open(configFile)
    # and read it into a list of strings
    configLines = config.readlines()

    # loop over the lines
    for line in configLines:
        # check if comment or empty
        if '#' in line or re.match('^\n', line):
            print '\nnot using this:', line.strip()
        # check if right number of commands are parsed
        elif len(line.strip().split()) != 12:
            print '\nwrong number of commands '
        else:
            print '\nCommand OK! '
            commandString = line.strip().split()
            maskPath = commandString[0]
            maskName = commandString[1]
            funcPath = commandString[2]
            funcRelPath = commandString[3]
            funcName = commandString[4]
            subjectList = commandString[5]
            outputPath = commandString[6]
            nodeMode = int(commandString[7])
            estParameters = int(commandString[8])
            givenC = float(commandString[9])
            givenE = float(commandString[10])
            conditionName = commandString[11]
            print 'maskPath =', maskPath
            print 'maskName =', maskName
            print 'funcPath =', funcPath
            print 'funcRelPath =', funcRelPath
            print 'funcName =', funcName
            print 'subjectList =', subjectList
            print 'outputPath =', outputPath
            print 'nodeMode? =', nodeMode
            print 'estimate Parameters? =', estParameters
            print 'given C =', givenC
            print 'given E =', givenE
            print 'fileExtenstion =', conditionName
            Wrapper(maskPath,
                    maskName,
                    funcPath,
                    funcRelPath,
                    funcName,
                    subjectList,
                    outputPath,
                    nodeMode,
                    estParameters,
                    givenC,
                    givenE,
                    conditionName,
                    numberCores,
                    saveFiles=saveFiles)

    return

# Boilerplate to call main():
if __name__ == '__main__':
    print 'Running in direct Call mode!'
    configFile = sys.argv[1]
    if len(sys.argv) > 2:
        numberCores = int(sys.argv[2])
        Main(configFile, numberCores)
    else:
        Main(configFile)