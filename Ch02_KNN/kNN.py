'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
'''
import numpy as np
import operator
from os import listdir

'''
输入参数：用于分类的输入向量是inX，
输入的训练样本集为dataSet， 标签向量为labels，
最后的参数k表示用于选择最近邻居的数目
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #计算距离用欧氏距离公式 sqrt(( xA 0 - xB 0 )^2 + ( xA 1 - xB 1 ) ^2)
    #输入样本和每个dataSet的差
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #平方
    sqDiffMat = diffMat**2
    #平方和
    sqDistances = sqDiffMat.sum(axis=1)
    #开方
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #使用itemgetter， 按照第二个元素的次序对元组进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#创建数据集和标签
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def file2matrix(filename):
    love_dictionary = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    #打开文件，得到文件的行
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #get the number of lines in the file
    #创建以零填充的矩阵NumPy
    returnMat = np.zeros((numberOfLines, 3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    index = 0
    for line in arrayOLines:
        #使用函数line.strip()截取掉所有的回车字符
        line = line.strip()
        #使用tab字符\t将上一步得 到的整行数据分割成一个元素列表
        listFromLine = line.split('\t')
        #选取前3个元素，将它们存储到特征矩阵中
        returnMat[index, :] = listFromLine[0:3]
        #将列表的最后一列存储到向量classLabelVector中
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else: #将字符串转换成树值
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

#归一化特征值
def autoNorm(dataSet):
    #每列的最小值放在变量 minVals 中 ,dataSet.min(0)中的参数0使得函数可以从列中选取最小值，而不是选取当前行的最小值
    minVals = dataSet.min(0)
    #将最大值放在变量 maxVals中
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    #数据的每一行都减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #在除以range 具体特征值相除
    normDataSet = normDataSet/np.tile(ranges, (m, 1))   #element wise divide
    return normDataSet, ranges, minVals
    #对于某些数值处理软件包，/可能意味着矩阵除法，但在NumPy库中，矩阵除法需要使用函数 linalg.solve(matA,matB)。

#对训练集预测
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input(\
                                  "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - \
                                  minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: %s" % resultList[classifierResult - 1])

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
