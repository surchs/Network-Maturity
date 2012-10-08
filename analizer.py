'''
Created on Oct 1, 2012

@author: sebastian

This script is a quick hack to compare the prediction accuracies of different
networks and output visual representations of the analysis
'''
import sys
import gzip
import cPickle
import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages as pdf


def Main(loadFile):
    # first get the file in - this may throw an error if the class of the
    # analysis object is not in the current PYTHONPATH - I'll find a solution
    # for this later
    openFile = gzip.open(loadFile, 'rb')
    analysis = cPickle.load(openFile)
    aName = analysis.name

    # loop over the networks and store their information in a DICT
    valueDict = {}
    shappStore = np.array([])
    errList = []
    maeList = []
    normCount = 0
    networks = analysis.networks
    networkNames = networks.keys()
    networkNames.sort()

    for networkName in networkNames:
        # all the values are stored in another DICT
        tempDict = {}
        tempNet = networks[networkName]
        tempTrue = tempNet.trueData
        tempPred = tempNet.predictedData
        tempErr = tempTrue - tempPred
        # append error to errorlist for ANOVA
        errList.append(tempErr)
        tempAbs = np.absolute(tempErr)
        tempMae = np.mean(tempAbs)
        # append mae to maelist for display
        maeList.append(tempMae)
        tempStd = np.std(tempErr)
        # get the p value of the shapiro-wilk test
        tempShapp = st.shapiro(tempErr)[1]
        if tempShapp >= 0.05:
            normCount += 1
        shappStore = np.append(shappStore, tempShapp)
        # assign these values to the DICT
        tempDict['true'] = tempTrue
        tempDict['pred'] = tempPred
        tempDict['error'] = tempErr
        tempDict['abs'] = tempAbs
        tempDict['std'] = tempStd
        tempDict['shapp'] = tempShapp
        tempDict['mae'] = tempMae
        # put the dictionary in the valueDict
        valueDict[networkName] = tempDict

    # now run the tests to determine if we can the ANOVA not implemented yet
    if shappStore.max() >= 0.05:
        print 'All networks are nicely normally distributed'
        # now run the ANOVA thing - right now, we run just everything
        anova = st.f_oneway(*errList)
        print '\nANOVA has run'
        print ('Behold the amazing F of '
               + str(round(anova[0], 4))
               + ' and p '
               + str(round(anova[1], 4)))

    else:
        print 'not all networks are normally distributed'
        print (str(normCount)
               + ' out of '
               + str(len(networkNames))
               + ' networks are normally distributed')

    '''
    now make with the visualization

    as a reminder: these are the figures we are using
        1) Boxplots of the network-specific distributions of raw errors
        2) Plot of raw error over true age
        3) Plot of absolute error over true age
        4) Plot of predicted age over true age with MAE in the legend

    now go and prepare for this
    '''

    numberNetworks = len(networkNames)

    '''
    I am taking this part out of the code because I only want two lines
    of plots:

    edge = np.sqrt(numberNetworks)

    if np.ceil(edge) == edge:
        print 'how nice, all', str(numberNetworks), 'networks fit in '
        rows = int(edge)
        cols = int(edge)
    else:
        print 'nah, not all', str(numberNetworks), 'networks are going to fit '
        rows = int(np.ceil(edge))
        cols = int(np.ceil(edge))
        leftOver = rows * cols - numberNetworks
        print str(leftOver), 'subplots will be left empty '

    '''

    # now cols are hardcoded and rows depend on them
    cols = 2.0
    rows = np.ceil(numberNetworks / cols)

    fig1 = plt.figure(1)
    fig1.suptitle('boxplots of error variance')
    # fig1.tight_layout()

    fig2 = plt.figure(2, figsize=(8.5, 11), dpi=150)
    fig2.suptitle('error over true age')
    # fig2.tight_layout()

    fig3 = plt.figure(3, figsize=(8.5, 11), dpi=150)
    fig3.suptitle('absolute error over true age')
    # fig3.tight_layout()

    fig4 = plt.figure(4, figsize=(8.5, 11), dpi=150)
    fig4.suptitle('predicted over true age')
    # fig4.tight_layout()

    fig5 = plt.figure(5)
    fig5.suptitle('mean absolute error of the networks')

    loc = 1

    errorVarList = []
    errorNameList = []
    # now loop over the networks and get the data
    for networkName in networkNames:
        # we'll start with a simple figure for all the networks
        tD = valueDict[networkName]

        # for the boxplots, we have to append the data to a list
        errorVarList.append(tD['error'])
        errorNameList.append(networkName)

        tSP2 = fig2.add_subplot(rows, cols, loc, title=networkName)
        tSP2.plot(tD['true'], tD['error'], 'co')

        tSP3 = fig3.add_subplot(rows, cols, loc, title=networkName)
        tSP3.plot(tD['true'], tD['abs'], 'co')

        tSP4 = fig4.add_subplot(rows, cols, loc, title=networkName)
        tSP4.plot(tD['true'], tD['true'])
        tSP4.plot(tD['true'], tD['pred'], 'co')
        # add 1 to the localization variable
        loc += 1

    # now we can build figure 1
    tSP1 = fig1.add_subplot(111)
    tSP1.boxplot(errorVarList)
    plt.setp(tSP1, xticklabels=errorNameList)

    # and now we build figure 5
    tSP5 = fig5.add_subplot(111)
    indMae = range(len(maeList))
    tSP5.bar(indMae, maeList, facecolor='#99CCFF', align='center')
    tSP5.set_ylabel('MAE for network')
    tSP5.set_xticks(indMae)
    # set x-labels to the network names
    tSP5.set_xticklabels(networkNames)
    fig5.autofmt_xdate()

    # adjust the images
    fig1.subplots_adjust(hspace=0.5, wspace=0.5)
    fig2.subplots_adjust(hspace=0.5, wspace=0.5)
    fig3.subplots_adjust(hspace=0.5, wspace=0.5)
    fig4.subplots_adjust(hspace=0.5, wspace=0.5)
    fig5.subplots_adjust(hspace=0.5, wspace=0.5)

    # now save all that to a pdf
    pp = pdf((aName + '_results.pdf'))
    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.savefig(fig3)
    pp.savefig(fig4)
    pp.savefig(fig5)
    pp.close()

    print '\nDone saving. Have a nice day.'

    pass


if __name__ == '__main__':
    loadFile = sys.argv[1]
    Main(loadFile)
    pass
