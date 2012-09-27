'''
Created on Jul 12, 2012

@author: sebastian

This script will be used to wrap around the classes in the processing package
'''
import os
import inspect
import sys
import cPickle
import gzip
import multiprocessing as mp

# realpath() will make your script run, even if you symlink it :)
currentFolder = inspect.getfile(inspect.currentframe())
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(currentFolder)[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
import preprocessing as ps


def tempWrap(subject):
    # temporary solution for multiprocessing inside the classes
    # the subject object gets handed to this thing from the analysis
    # (over which we have to loop if there is more than one) and gets executed
    # inside this function. Subsequently, we have to map it back to its
    # analysis according to its name

    subject.getTimeseries()
    subject.getConnectivity()
    subject.getFeatures()
    print 'Done with subject', subject.name
    return subject


def Main(configFile, studyName, outPath, numberCores):
    # this function is used to get the files and then continue and call all the
    # necessary methods inside the Study class

    studyParameter = ps.parameters('Network Maturity')

    studyParameter.makeStudy(configFile, outPath, numberCores)
    # this has taken care of the parameter stuff. Now we have to get the Study

    Study = ps.study(studyParameter)
    # From now on we just have to call the methods of the Study
    #
    # load the Parameters of that are given in the config file

    Study.getParameters()
    # make these Parameters into preproc

    Study.makePreproc()
    # initialize all the preproc

    Study.initializePreproc()
    # and finally run the preprocessing

    # Study.preprocessAnalysis()

    # get the preproc from the study object
    Analyses = Study.givePreproc()

    for analysis in Analyses.keys():
        tempAnalysis = Analyses[analysis]
        argList = []
        print 'Running Analysis', analysis, 'now!'
        for subject in tempAnalysis.subjects.keys():
            tempSubject = tempAnalysis.subjects[subject]
            argList.append(tempSubject)
        print 'Setup arguments for', analysis, '!'
            # we can just assume that the subject name and the analysis key are
            # the same, otherwise this would have to be passed
        # run the current analysis in parallel
        pool = mp.Pool(processes=numberCores)
        print 'declared pools for', analysis, '!'
        resultList = pool.map(tempWrap, argList)
        pool.close()
        print 'Done running analysis', analysis, 'now!'
        # now map the shit back
        print 'Begin mapping back the results'
        for result in resultList:
            tempSubject = result
            subName = tempSubject.name
            tempAnalysis.subjects[subName] = tempSubject
        Study.preproc[analysis] = tempAnalysis
        # save the study object
        print 'Done mapping back the results'
        # store and get rid of the analysis
        Study.saveThis('Preprocessing', analysis)

    print 'Wrapper: initializing the analyses'
    Study.initializeAnalysis()
    # don't need this, more efficient cleanup in place
    #Study.cleanup(1)

    Study.runAnalysis()
    # and lastly we ask the study to clean up as we don't want the file to get
    # too big - we don't need this at the moment
    # Study.cleanup(2)

    # so far this should be all, now we just have to save the Study object

    output = gzip.open(studyName, 'wb')
    cPickle.dump(Study, output, protocol=2)
    print 'Done saving the study!'
    print 'Have a nice day'
    # all done


if __name__ == '__main__':
    # get the files that are passed to the file
    print 'Runing in direct call mode'
    configFile = sys.argv[1]
    studyName = sys.argv[2]
    outPath = sys.argv[3]
    if len(sys.argv) > 4:
        numberCores = int(sys.argv[4])
    else:
        # if no commands are given, we use 4 cores
        numberCores = 4
    # call the Main
    Main(configFile, studyName, outPath, numberCores)
    pass
