# coding: utf-8
from numpy import *
import pandas as pd
import codecs
import operator
from math import log
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
    for i in range(numFeatures):
        # 对连续型特征进行处理 ,i代表第i个特征,featList是每次选取一个特征之后这个特征的所有样本对应的数据
        featList = [example[i] for example in dataSet]

        uniqueVals = set(featList)
        newGini = 0.0
        # 计算该特征下每种划分的信息熵,选取第i个特征的值为value的子集
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            print prob
            newGini += prob * calcGini(subDataSet)

        print u"label:{0} Gini:{1}".format(labels[i],newGini)
        if newGini < bestGini:
            bestGini = newGini
            bestFeature = i

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

def createTree(dataSet,labels,data_full,labels_full):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):  #注释1
        return classList[0]
    if len(dataSet[0])==1:                             #注释2
        return majorityCnt(classList)
    #平凡情况，每次找到最佳划分的特征
    bestFeat=chooseBestFeatureToSplit(dataSet,labels)
    bestFeatLabel=labels[bestFeat]

    myTree={bestFeatLabel:{}}
    featValues=[example[bestFeat] for example in dataSet]
    '''
    刚开始很奇怪为什么要加一个uniqueValFull，后来思考下觉得应该是在某次划分，比如在根节点划分纹理的时候，将数据分成了清晰、模糊、稍糊三块
    ，假设之后在模糊这一子数据集中，下一划分属性是触感，而这个数据集中只有软粘属性的西瓜，这样建立的决策树在当前节点划分时就只有软粘这一属性了，
    事实上训练样本中还有硬滑这一属性，这样就造成了树的缺失，因此用到uniqueValFull之后就能将训练样本中有的属性值都囊括。
    如果在某个分支每找到一个属性，就在其中去掉一个，最后如果还有剩余的根据父节点投票决定。
    但是即便这样，如果训练集中没有出现触感属性值为“一般”的西瓜，但是分类时候遇到这样的测试样本，那么应该用父节点的多数类作为预测结果输出。
    '''
    uniqueVals=set(featValues)
    currentlabel=labels_full.index(labels[bestFeat])
    featValuesFull=[example[currentlabel] for example in data_full]
    uniqueValsFull=set(featValuesFull)
    del(labels[bestFeat])

    '''
    针对bestFeat的每个取值，划分出一个子树。对于纹理，树应该是{"纹理"：{？}}，显然？处是纹理的不同取值，有清晰模糊和稍糊三种，对于每一种情况，
    都去建立一个自己的树，大概长这样{"纹理"：{"模糊"：{0},"稍糊"：{1},"清晰":{2}}}，对于0\1\2这三棵树，每次建树的训练样本都是值为value特征数减少1
    的子集。
    '''

    for value in uniqueVals:
        subLabels = labels[:]
        uniqueValsFull.remove(value)
        myTree[bestFeatLabel][value] = createTree(splitDataSet \
                            (dataSet, bestFeat, value), subLabels, data_full, labels_full)


    for value in uniqueValsFull:
        myTree[bestFeatLabel][value] = majorityCnt(classList)
    return myTree

if __name__=="__main__":
    # 读入csv文件数据
    input_path = "data/西瓜数据集2.0.csv"
    file = codecs.open(input_path, "r", 'utf-8')

    filedata = [line.strip('\n').split(',') for line in file]
    filedata = [[float(i) if '.' in i else i for i in row] for row in filedata]  # change decimal from string to float
    dataSet = [row[1:] for row in filedata[1:11]]
    print len(dataSet)
    data_full = dataSet[:]
    labels = []
    for label in filedata[0][1:-1]:
        labels.append(label)
    labels_full = labels[:]
    myTree = createTree(dataSet, labels, data_full, labels_full)
    createPlot(myTree)
    print json.dumps(myTree, ensure_ascii=False, indent=4)



    # df = pd.read_csv(u'data/西瓜数据集2.0.csv')
    # data = df.values[:, 1:].tolist()
    # data_full = data[:]
    # labels = df.columns.values[1:-1].tolist()
    # labels_full = labels[:]
    #
    # myTree = createTree(data, labels, data_full, labels_full)
    # # createPlot(myTree)
    # print json.dumps(myTree, ensure_ascii=False, indent=4)




