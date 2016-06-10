# -*- coding: utf-8 -*-
'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
#----------------------------------------------------- 
#   功能：机器学习第五章示例代码
#   作者：Peter
#   TODO：进行少量代码的修改，但在82页绘图时与原作有一定差异。 
#   学习时间：2016-03-14  
#   语言：Python 2.7.6  
#   环境：linux（ubuntu14.04）
#-----------------------------------------------------
'''
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

#任务1：给定两类点，找到一条直线分割平面********************************************************************************
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
#自己练习
def loadDataSet1():
    fr=  open('testSet.txt','r')
    lineArr = fr.readlines()
    lineArr = [line.strip().split('\t') for line in lineArr]
    DataArr = [[1.0,line[0],line[1]]for line in lineArr]
    labelMat = [ line[2] for line in lineArr]
    return DataArr,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))
#权重从全1开始，然后每一次迭代将所有样本计算一次得到一个错误矩阵（维度和输出的类别矩阵一样），然后根据样本矩阵和错误矩阵的乘积来更新权重
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix

    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 200
    weights = ones((n,1))
    saved = []				
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
        saved.append(weights)
    #为了便于比较几种优化方法之间得到的权重之间的差异，统一将权重的第一位设置为1
    weights = weights/weights[0]
    print "type of weights:{0}".format(type(weights))
    print "weights:{0}".format(weights)
    return weights, saved

def stocGradAscent0(dataMatrix, classLabels,numIter = 150):

    m,n = shape(dataMatrix)
    alpha = 0.5
    saved=[]				    
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):				
        dataIndex = range(m)
        for i in range(m):
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant									
            h = sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
            saved.append(weights)								   
            del(dataIndex[randIndex])
    #为了便于比较几种优化方法之间得到的权重之间的差异，统一将权重的第一位设置为1
    weights = weights / weights[0]
    print "type of weights:{0}".format(type(weights))
    print "weights:{0}".format(weights)
    return weights, array(saved)

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    saved = []
    alpha = 0.4				
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            saved.append(weights)												
            del(dataIndex[randIndex])
    #为了便于比较几种优化方法之间得到的权重之间的差异，统一将权重的第一位设置为1
    weights = weights / weights[0]
    print "type of weights:{0}".format(type(weights))
    print "weights:{0}".format(weights)
    return weights,array(saved)

#输入weights的格式要求为数组
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    if type(weights)!=type(np.array(1)):
        weights = weights.getA()
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    #函数原型是0=w[0]+w[1]*x[1]+w[2]*x[2]，每取定一个x当做x[1]，经过逆推得到y=x2 = (-w[0]-w[1]*x[1])/w[2]
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
#此函数是我自己加的，目的是绘制训练过程中参数随着迭代次数的变化，但是有一点疑惑
#就是绘制出来的图形和作者的原图像差距较大，尚不明白错误发生在什么地方				
def plotTraining(para):#输入para的格式要求为数组
    import matplotlib.pyplot as plt
    #weights = para.getA()
    iterNumber = len(para)				
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    x1 = arange(iterNumber)
    y1 = para[:,0]
    ax1.plot(x1,y1)
    plt.xlabel('iternumber')
    plt.ylabel('X0')
    ax2 = fig.add_subplot(312)
    x2 = arange(iterNumber)
    y2 = para[:,1]
    ax2.plot(x2,y2)
    plt.xlabel('iternumber')
    plt.ylabel('X1')
    ax3 = fig.add_subplot(313)
    x3 = arange(iterNumber)
    y3 = para[:,2]
    ax3.plot(x3,y3)
    plt.xlabel('iternumber')
    plt.ylabel('X2')				
    plt.show()

#任务2：从一些病症判断马的死亡率，一共有20个特征并且存在一定的缺失值************************************************************
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    print trainingSet[0],trainingLabels[0]
    trainWeights,saved = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # print trainWeights.shape
        # print int(classifyVector(array(lineArr), trainWeights))
        # print currLine[21]
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest(num_tests):
    numTests = 10
    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))



if __name__ == '__main__':
    #任务1：给定两类点，找到一条直线分割平面
    #dataArr, labelMat = loadDataSet()
    #批梯度下降法
    #weights, saved = gradAscent(dataArr, labelMat)
    #随机梯度下降法
    #weights,saved = stocGradAscent0(array(dataArr),labelMat,200)
    #改进版的随机梯度下降
    #weights,saved = stocGradAscent1(array(dataArr),labelMat,150)
    #绘制训练得到的最佳拟合直线
    #plotBestFit(weights)
    #绘制训练参数图
    #plotTraining(saved)

    #任务2：从一些病症判断马的死亡率，一共有20个特征并且存在一定的缺失值
    #单次测试
    #colicTest()
    #多次测试，设置测试的次数
    multiTest(10)