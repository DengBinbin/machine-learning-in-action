# -*- coding: utf-8 -*-
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin

#deng modify the code in 2016.03.09 
'''
from numpy import *
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)		
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
       
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1						
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
  			
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
  
def classifyPerson():#written by deng
    resultList=['not at all','in small doses','in large doses']
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')	
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print normMat.shape,ranges.shape,minVals.shape
    VideoGameTime=float(raw_input("percentage of time playing video game?"))
    FilerMiles=float(raw_input("frequent flier miles earned per year?"))
    IceCream=float(raw_input("liters of ice cream consumed per year?"))
    personInfo1=array([FilerMiles,VideoGameTime,IceCream])
  			
    ClassifyResult=classify0((personInfo1-minVals)/ranges,normMat,datingLabels,5)
    print ClassifyResult				
    print "you will probably like this person:%s" %(resultList[ClassifyResult-1])				
				
				
				
				
				
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():#written by deng
    hwLabels=[]
    trainingList=listdir('digits/trainingDigits')#the folder of training set
    m=len(trainingList)				
    trainingMat = zeros((m,1024))				
  
    for i in range(m):
        trainingNameStr = trainingList[i]
        trainingName = trainingList[i].split('.')[0]
        trainingLabels = int(trainingName.split('_')[0])
        hwLabels.append(trainingLabels)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' %trainingNameStr)
    testList=listdir('digits/testDigits')#the folder of training set
    mTest=len(testList)
    errorCount=0.0				
    for i in range(mTest):
        testNameStr = testList[i]
        testName = testList[i].split('.')[0]
        Labels = int(testName.split('_')[0])
        testVec = img2vector('digits/testDigits/%s' %testNameStr)									    
        trainingResult=classify0(testVec,trainingMat,hwLabels,3)
        print "the classifier came back with %d ,the real answer is %d" %(trainingResult,Labels)        					
        if(trainingResult!=Labels): errorCount+=1							
    print "\nthe total number of training is %d" %m									
    print "\nthe total number of test is %d" %mTest									
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
				
group,labels=createDataSet()				
datingDataMat,datingLabels= file2matrix('datingTestSet2.txt')
normMat, ranges, minVals = autoNorm(datingDataMat)
testVector=img2vector('digits/testDigits/0_13.txt')
