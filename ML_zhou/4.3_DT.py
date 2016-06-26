# coding: utf-8
from numpy import *
import pandas as pd
import codecs
import operator
from math import log
from treePlotter import *
import matplotlib.pyplot as plt


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)

    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

#将数据按照axis的值为value去进行划分
def splitDataSet(dataSet,axis,value):
    returnMat = []
    for data in dataSet:
        if data[axis]==value:
            returnMat.append(data[:axis]+data[axis+1:])
    return returnMat

'''
如何选取最佳的切分点呢？一个很显然的思路，对于数据可能的切分方式，每一种都计算一次熵降，看看哪个最大就选择哪个
params:base_ent原始切分前的数据集的熵
ent_des熵降
split_ent切分数据集之后的熵
'''
def chooseBestFeatureToSplit(dataSet):
    base_ent = calcShannonEnt(dataSet)

    num_features = len(dataSet[0])-1
    infoGain = 0
    ent_des = 0
    bestFeature = -1
    for i in range(num_features):
        feature_list = [example[i] for example in dataSet]
        #上面求得的特征表是有重复特征的，只需求出其中不同元素即可
        feature_list = set(feature_list)
        split_ent = 0
        for value in feature_list:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            split_ent+=prob*calcShannonEnt(subDataSet)
        ent_des = base_ent-split_ent

        if ent_des>infoGain:
            infoGain = ent_des
            bestFeature = i
    return bestFeature

'''
这个函数的作用是返回字典中出现次数最多的value对应的key，也就是输入list中出现最多的那个值
'''
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

'''
递归地去建树
'''
def createTree(dataSet,labels):
    classList = [example[-1]for example in dataSet]
    #classList:['yes', 'yes', 'no', 'no', 'no']
    #如果classList里面全部都是它的第0个元素，即全部分为1类了
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #len(dataSet[0])是输入数据的列数，因为有一列是label，所以当len(dataSet)等于1的时候说明所有属性
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #一般情况
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    #每次找到的最佳切分点之后，按照最佳切分点
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

# 读入csv文件数据
input_path = "data/西瓜数据集3.csv"
file = codecs.open(input_path,"r",'utf-8')

filedata = [line.strip('\n').split(',') for line in file]
#filedata = [[float(i) if '.' in i else i for i in row ] for row in filedata] # change decimal from string to float
dataSet = [row[1:] for row in filedata[1:]]

labels = []
for label in filedata[0][1:-1]:
    labels.append(label)

myTree = createTree(dataSet,labels)
createPlot(myTree)