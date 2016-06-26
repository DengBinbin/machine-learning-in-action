#coding=utf-8
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
deng change some code for learning ML

'''
from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

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

'''
首先找到根节点，由根节点就可以得到根节点底下的二级字典。输入数据在根节点属性的值是0还是1就可以得到当根节点做出预测的时候的valueOfFeat，这个时候
还需要进行一次判断，如果这个时候还是字典，说明没有分到底部，则需要继续划分，否则直接返回valueOfFeat
'''
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print valueOfFeat
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


#作者的代码
# def splitDataSet(dataSet, axis, value):
#     retDataSet = []
#     for featVec in dataSet:
#         if featVec[axis] == value:
#             reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
#             reducedFeatVec.extend(featVec[axis+1:])
#             retDataSet.append(reducedFeatVec)
#     return retDataSet
#
# def chooseBestFeatureToSplit(dataSet):
#     numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
#     baseEntropy = calcShannonEnt(dataSet)
#     bestInfoGain = 0.0; bestFeature = -1
#     for i in range(numFeatures):        #iterate over all the features
#         featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
#         uniqueVals = set(featList)       #get a set of unique values
#         newEntropy = 0.0
#         for value in uniqueVals:
#             subDataSet = splitDataSet(dataSet, i, value)
#             prob = len(subDataSet)/float(len(dataSet))
#             newEntropy += prob * calcShannonEnt(subDataSet)
#         infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
#         if (infoGain > bestInfoGain):       #compare this to the best gain so far
#             bestInfoGain = infoGain         #if better than current best, set to best
#             bestFeature = i
#
#     return bestFeature                      #returns an integer

# def createTree(dataSet,labels):
#     classList = [example[-1] for example in dataSet]
#     if classList.count(classList[0]) == len(classList):
#         return classList[0]#stop splitting when all of the classes are equal
#     if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
#         return majorityCnt(classList)
#     bestFeat = chooseBestFeatureToSplit(dataSet)
#     bestFeatLabel = labels[bestFeat]
#     myTree = {bestFeatLabel:{}}
#     del(labels[bestFeat])
#     featValues = [example[bestFeat] for example in dataSet]
#     uniqueVals = set(featValues)
#     for value in uniqueVals:
#         subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
#         myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
#     return myTree

#
#
# def classify(inputTree,featLabels,testVec):
#     firstStr = inputTree.keys()[0]
#     secondDict = inputTree[firstStr]
#     featIndex = featLabels.index(firstStr)
#     key = testVec[featIndex]
#     valueOfFeat = secondDict[key]
#     if isinstance(valueOfFeat, dict):
#         classLabel = classify(valueOfFeat, featLabels, testVec)
#     else: classLabel = valueOfFeat
#     return classLabel
#
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]



if __name__ =="__main__":
    myDat, labels = createDataSet()

    #bestFeature=chooseBestFeatureToSplit(myDat)

    #myTree = createTree(myDat, labels)

    myTree = retrieveTree(0)

    print classify(myTree,labels,[1,1])