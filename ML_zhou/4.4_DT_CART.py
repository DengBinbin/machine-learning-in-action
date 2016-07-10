# coding: utf-8
from numpy import *
import pandas as pd
import codecs
import operator
import copy
import json
from treePlotter import *


'''
输入：给定数据集   输出：Gini指数

'''
def calcGini(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Gini = 1.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        Gini -= prob * prob #log base 2
    return Gini
'''
输入：数据集，划分特征，划分特征的取值         输出：划分完毕后的数据子集
这个函数的作用是对数据集进行划分，对属性axis值为value的那部分数据进行挑选（注：此处得到的子集特征数比划分前少1，少了那个用来划分的数据）
'''
def splitDataSet(dataSet,axis,value):
    returnMat = []
    for data in dataSet:
        if data[axis]==value:
            returnMat.append(data[:axis]+data[axis+1:])
    return returnMat

'''
与上述函数类似，区别在于上述函数是用来处理离散特征值而这里是处理连续特征值
对连续变量划分数据集，direction规定划分的方向，
决定是划分出小于value的数据样本还是大于value的数据样本集
'''
def splitContinuousDataSet(dataSet, axis, value, direction):
    retDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:
                retDataSet.append(featVec[:axis] + featVec[axis + 1:])
        else:
            if featVec[axis] <= value:
                retDataSet.append(featVec[:axis] + featVec[axis + 1:])
    return retDataSet

'''
决策树算法中比较核心的地方，究竟是用何种方式来决定最佳划分？
使用信息增益作为划分标准的决策树称为ID3
使用信息增益比作为划分标准的决策树称为C4.5
本题为信息增益的ID3树
从输入的训练样本集中，计算划分之前的熵，找到当前有多少个特征，遍历每一个特征计算信息增益，找到这些特征中能带来信息增益最大的那一个特征。
这里用分了两种情况，离散属性和连续属性
1、离散属性，在遍历特征时，遍历训练样本中该特征所出现过的所有离散值，假设有n种取值，那么对这n种我们分别计算每一种的熵，最后将这些熵加起来
就是划分之后的信息熵
2、连续属性，对于连续值就稍微麻烦一点，首先需要确定划分点，用二分的方法确定（连续值取值数-1）个切分点。遍历每种切分情况，对于每种切分，
计算新的信息熵，从而计算增益，找到最大的增益。
假设从所有离散和连续属性中已经找到了能带来最大增益的属性划分，这个时候是离散属性很好办，直接用原有训练集中的属性值作为划分的值就行，但是连续
属性我们只是得到了一个切分点，这是不够的，我们还需要对数据进行二值处理。
'''
def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0]) - 1
    bestGini = 10000.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        # 对连续型特征进行处理 ,i代表第i个特征,featList是每次选取一个特征之后这个特征的所有样本对应的数据
        featList = [example[i] for example in dataSet]
        # 因为特征分为连续值和离散值特征，对这两种特征需要分开进行处理。
        if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int':
            # 产生n-1个候选划分点
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)
            bestSplitGini = 10000
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for value in splitList:
                newGini = 0.0
                subDataSet0 = splitContinuousDataSet(dataSet, i, value, 0)
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(subDataSet0) / float(len(dataSet))
                newGini += prob0 * calcGini(subDataSet0)
                prob1 = len(subDataSet1) / float(len(dataSet))
                newGini += prob1 * calcGini(subDataSet1)
                if newGini < bestSplitGini:
                    bestSplitGini = newGini
                    bestSplit = value
                    # 用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]] = bestSplit
            newGini = bestSplitGini
        else:
            uniqueVals = set(featList)
            newGini = 0.0
            # 计算该特征下每种划分的信息熵,选取第i个特征的值为value的子集
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newGini += prob * calcGini(subDataSet)
        if newGini < bestGini:
            bestGini = newGini
            bestFeature = i

    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理
    # 即是否小于等于bestSplitValue
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0
    return bestFeature

'''
输入：类别列表     输出：类别列表中多数的类，即多数表决
这个函数的作用是返回字典中出现次数最多的value对应的key，也就是输入list中出现最多的那个值
'''
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]



# 由于在Tree中，连续值特征的名称以及改为了feature <= value的形式
# 因此对于这类特征，需要利用正则表达式进行分割，获得特征名以及分割阈值
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    if u'<=' in firstStr:
        featvalue = float(firstStr.split(u"<=")[1])
        featkey = firstStr.split(u"<=")[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(featkey)
        if testVec[featIndex] <= featvalue:
            judge = 1
        else:
            judge = 0
        for key in secondDict.keys():
            if judge == int(key):
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
    else:
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
    return classLabel


def testing(myTree, data_test, labels):
    error = 0.0
    for i in range(len(data_test)):
        if classify(myTree, labels, data_test[i]) != data_test[i][-1]:
            error += 1
    # print 'myTree %d' % error
    return float(error)

def testing_feat(feat,train_data,test_data,labels):
    class_list = [example[-1] for example in train_data]
    bestFeatIndex = labels.index(feat)
    train_data = [example[bestFeatIndex] for example in train_data]
    test_data = [(example[bestFeatIndex],example[-1]) for example in test_data]
    all_feat = set(train_data)
    error = 0.0
    for value in all_feat:
        class_feat = [ class_list[i] for i in range(len(class_list)) if train_data[i]==value]
        major = majorityCnt(class_feat)
        for data in test_data:
            if data[0]==value and data[1]!=major:
                error+=1.0
    # print 'myTree %d' % error
    return error

def testingMajor(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1
    # print 'major %d' % error
    return float(error)

# #后剪枝
# def postPruningTree(inputTree,dataSet,data_test,labels):
#     firstStr=inputTree.keys()[0]
#     secondDict=inputTree[firstStr]
#     classList=[example[-1] for example in dataSet]
#     featkey=copy.deepcopy(firstStr)
#     if u'<=' in firstStr:
#         featkey=firstStr.split(u'<=')[0]
#         featvalue=float(firstStr.split(u'<=')[1])
#     labelIndex=labels.index(featkey)
#     temp_labels=copy.deepcopy(labels)
#     del(labels[labelIndex])
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__=='dict':
#             if type(dataSet[0][labelIndex]).__name__=='unicode':
#                 inputTree[firstStr][key]=postPruningTree(secondDict[key],\
#                  splitDataSet(dataSet,labelIndex,key),splitDataSet(data_test,labelIndex,key),copy.deepcopy(labels))
#             else:
#                 inputTree[firstStr][key]=postPruningTree(secondDict[key],\
#                 splitContinuousDataSet(dataSet,labelIndex,featvalue,key),\
#                 splitContinuousDataSet(data_test,labelIndex,featvalue,key),\
#                 copy.deepcopy(labels))
#     if testing(inputTree,data_test,temp_labels)<=testingMajor(majorityCnt(classList),data_test):
#         return inputTree
#     return majorityCnt(classList)



'''
主程序，递归产生决策树。
params:
dataSet:用于构建树的数据集,最开始就是data_full，然后随着划分的进行越来越小，第一次划分之前是17个瓜的数据在根节点，然后选择第一个bestFeat是纹理
纹理的取值有清晰、模糊、稍糊三种，将瓜分成了清晰（9个），稍糊（5个），模糊（3个）,这个时候应该将划分的类别减少1以便于下次划分
labels：还剩下的用于划分的类别
data_full：全部的数据
label_full:全部的类别
既然是递归的构造树，当然就需要终止条件，终止条件有三个：
1、当前节点包含的样本全部属于同一类别；-----------------注释1就是这种情形
2、当前属性集为空，即所有可以用来划分的属性全部用完了，这个时候当前节点还存在不同的类别没有分开，这个时候我们需要将当前节点作为叶子节点，
同时根据此时剩下的样本中的多数类（无论几类取数量最多的类）-------------------------注释2就是这种情形
3、当前节点所包含的样本集合为空。比如在某个节点，我们还有10个西瓜，用大小作为特征来划分，分为大中小三类，10个西瓜8大2小，因为训练集生成
树的时候不包含大小为中的样本，那么划分出来的决策树在碰到大小为中的西瓜（视为未登录的样本）就会将父节点的8大2小作为先验同时将该中西瓜的
大小属性视作大来处理。
'''

def createTree(dataSet,labels,data_full,labels_full,test_data,mode="unpro"):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):  #注释1
        return classList[0]
    if len(dataSet[0])==1:                             #注释2
        return majorityCnt(classList)
    #平凡情况，每次找到最佳划分的特征
    labels_copy = copy.deepcopy(labels)
    bestFeat=chooseBestFeatureToSplit(dataSet,labels)
    bestFeatLabel=labels[bestFeat]
    print bestFeatLabel
    if mode=="unpro" or mode=="post":
        myTree = {bestFeatLabel: {}}
    elif mode=="prev":
        if testing_feat(bestFeatLabel,dataSet,test_data,labels_copy)<testingMajor(majorityCnt(classList), test_data):
            myTree = {bestFeatLabel: {}}
        else:
            return majorityCnt(classList)

    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    '''
    刚开始很奇怪为什么要加一个uniqueValFull，后来思考下觉得应该是在某次划分，比如在根节点划分纹理的时候，将数据分成了清晰、模糊、稍糊三块
    ，假设之后在模糊这一子数据集中，下一划分属性是触感，而这个数据集中只有软粘属性的西瓜，这样建立的决策树在当前节点划分时就只有软粘这一属性了，
    事实上训练样本中还有硬滑这一属性，这样就造成了树的缺失，因此用到uniqueValFull之后就能将训练样本中有的属性值都囊括。
    如果在某个分支每找到一个属性，就在其中去掉一个，最后如果还有剩余的根据父节点投票决定。
    但是即便这样，如果训练集中没有出现触感属性值为“一般”的西瓜，但是分类时候遇到这样的测试样本，那么应该用父节点的多数类作为预测结果输出。
    '''
    if type(dataSet[0][bestFeat]).__name__ == 'unicode':

        currentlabel = labels_full.index(labels[bestFeat])
        featValuesFull = [example[currentlabel] for example in data_full]
        uniqueValsFull = set(featValuesFull)

    del(labels[bestFeat])

    '''
    针对bestFeat的每个取值，划分出一个子树。对于纹理，树应该是{"纹理"：{？}}，显然？处是纹理的不同取值，有清晰模糊和稍糊三种，对于每一种情况，
    都去建立一个自己的树，大概长这样{"纹理"：{"模糊"：{0},"稍糊"：{1},"清晰":{2}}}，对于0\1\2这三棵树，每次建树的训练样本都是值为value特征数减少1
    的子集。
    '''
    for value in uniqueVals:
        subLabels = labels[:]
        if type(dataSet[0][bestFeat]).__name__ == 'unicode':
            uniqueValsFull.remove(value)

        myTree[bestFeatLabel][value] = createTree(splitDataSet \
                            (dataSet, bestFeat, value), subLabels, data_full, labels_full,splitDataSet \
                            (test_data, bestFeat, value),mode=mode)
    if type(dataSet[0][bestFeat]).__name__ == 'unicode':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value] = majorityCnt(classList)

    if mode=="post":
        if testing(myTree, test_data, labels_copy) <= testingMajor(majorityCnt(classList), test_data):
            return myTree
    return majorityCnt(classList)


# 读入csv文件数据
def load_data(file_name):
    file = codecs.open(file_name, "r", 'utf-8')
    filedata = [line.strip('\n').split(',') for line in file]
    filedata = [[float(i) if '.' in i else i for i in row] for row in filedata]  # change decimal from string to float
    train_data = [row[1:] for row in filedata[1:12]]
    test_data = [row[1:] for row in filedata[12:]]
    labels = []
    for label in filedata[0][1:-1]:
        labels.append(unicode(label))
    return train_data,test_data,labels



if __name__=="__main__":

    train_data,test_data,labels = load_data("data/西瓜数据集2.0.csv")
    data_full = train_data[:]
    labels_full = labels[:]
    '''
    为了代码的简洁，将预剪枝，后剪枝和未剪枝三种模式用一个参数mode传入建树的过程
    post代表后剪枝，prev代表预剪枝，unpro代表不剪枝
    '''
    myTree = createTree(train_data,labels, data_full, labels_full,test_data,mode = "unpro")
    # myTree = postPruningTree(myTree,train_data,test_data,labels_full)
    # createPlot(myTree)
    print json.dumps(myTree, ensure_ascii=False, indent=4)






