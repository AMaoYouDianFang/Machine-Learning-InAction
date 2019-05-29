'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers'] #属性的名称分别对应第0个，第1个名称
    #change to discrete values
    return dataSet, labels

#计算给定数据集的信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #计算数据集中实例的总数
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        #当前样本的标签
        currentLabel = featVec[-1] 
        #统计每个label对应的样本数量
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #根据每个label计算信息熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2) #log base 2
    return shannonEnt

'''
功能：根据特征划分数据集 （划分后会去掉这个特征的）
内容：将对每个特征划分数 据集的结果计算一次信息熵，然后判断按照哪个特征划分数据集是最好的划分方式
参数：待划分的数据集、axis 第几个特征、value这个特征的取值，返回满足这个条件的数据
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = [] #为了不修改原始数据集，创建一个新的列表对象
    for featVec in dataSet:     #遍历样本，如果特征axis是value，那么将样本中的axis特征去掉
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #0- axis-1 的特征
            reducedFeatVec.extend(featVec[axis+1:]) #axis+1 到最后的特征 extend是连接
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式，返回最好的特征的下标
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #最后一列是label
    baseEntropy = calcShannonEnt(dataSet)  #所有数据的信息熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #遍历每个特征，找到最好的特征
        #将数据集中所有第i个特征值写入这个新list中
        featList = [example[i] for example in dataSet]
        #使用set去除重复元素
        uniqueVals = set(featList)      
        newEntropy = 0.0 
        for value in uniqueVals: #遍历特征i所有可能的取值，对每个唯一属性值划分一次数据集
            subDataSet = splitDataSet(dataSet, i, value) 
            prob = len(subDataSet)/float(len(dataSet)) 
            newEntropy += prob * calcShannonEnt(subDataSet) #计算条件熵
        infoGain = baseEntropy - newEntropy     #信息增益，越大效果越好
        if (infoGain > bestInfoGain):       
            bestInfoGain = infoGain         
            bestFeature = i                 #保存最好信息熵对应的索引
    return bestFeature                      #returns an integer

'''
递归构建决策树
原理:首先基于最好的属性值划分数据集，由于这个属性的取值可能多于两个，因此可能存在大于两个分支的数据集划分，
第一次划分之后，数据将被向下传递到树分支的下一个节点，在这个节点上，我们可以再次划分数据。因此我们
可以采用递归的原则处理数据集。

递归结束的条件是：1或者2
1. 程序遍历完所有划分数据集的属性(没有可以继续分解的属性) 
在算法开始运行前计算列的数目，查看算法是否使用了所有属性。如果数据集已经处理了所有属性，但是类标签依然不是唯一的，
通常会采用多数表决的方法决定该叶子节点的分类。

2. 每个分支下的所有实例都具有相同的分类。
如果所有实例具有相同的分类，则得到一个叶子节点或者终止块。任何到达叶子节点的数据必然属于叶子节点的分类，
'''

#投票表决
def majorityCnt(classList):  #label-- 票数 
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    #根据票数降序排列, 利用operator操作键值排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] #包含了数据集的所有类标签
    #递归结束的条件1：所有的类标签完全相同，则直接返回该类标签
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    #递归结束的条件2: 使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1: #只剩下标签列
        return majorityCnt(classList) #挑选出现次数最多的类别作为返回值
    bestFeat = chooseBestFeatureToSplit(dataSet) #选取最好的特征
    bestFeatLabel = labels[bestFeat]  #最好的特征对应的名称
    #创建树
    myTree = {bestFeatLabel:{}}  #两个dic嵌套
    del(labels[bestFeat])  #因为每次根据特征划分数据集后会去掉这个特征，所以要删除对应的名称
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues) #选取的最好的特征对应取值
    for value in uniqueVals:
        subLabels = labels[:]       #复制了类名称labels，并将其存储在新列表变量subLabels中
        #保障每次调用函数createTree()时不改变原始列表的内容，使用新变量subLabels代替原始列表
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#使用决策树执行分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree)[0]  #当前树节点的标签
    secondDict = inputTree[firstStr] #子节点
    featIndex = featLabels.index(firstStr) #将属性的名称转换为索引
    key = testVec[featIndex] #当前数据对这个属性的取值
    valueOfFeat = secondDict[key] #secondDict是当前属性分类后的结果，选择key对应的结果
    if isinstance(valueOfFeat, dict): #如果选择的结果valueOfFeat是dict，说明没有到达叶子节点需要继续分类
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat #如果是叶子节点，返回结果
    return classLabel

#在磁盘上保存训练好的树，使用Python模块pickle序列化对象
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
#读取保存的树
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

