'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

#函数读取一个以tab键为分隔符的文件，然后将每行的内容保存成一组浮点数
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat
'''
参数：数据集合、待切分的特征和该特征的某个值。
在给定特征和特征值的情况下，该函数通过数组过滤方式将上述数据集合切分得到两个子集并返回。
nonzero(a)  将对矩阵a的所有非零元素， 分别安装两个维度， 一次返回其在各维度上的目录值。
如果 a=mat([ [1,0,0],                          
             [1,0,0],
             [0,0,0]])                      
 则 nonzero(a)返回值为(array([0, 1]),array([0, 0])), 因为矩阵a只有两个非零值，在第0行、第0列，
和第1行、第0列。所以结果元组中，第一个行维度数据为（0,1）元组第二个列维度都为（0,0）。
'''
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]  #[0]返回nonzero的第一个维度，数据坐在的行
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1
'''
生成叶节点，当chooseBestSplit() 函数确定不再对数据进行切分时，
将调用该regLeaf()函数来得到叶节点的模型。在回归树中， 该模型其实就是目标变量的均值。
'''
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])
'''
误差估计函数regErr()。该函数在给定数据上计算目标变量的平方误差。
当然也可以先计算出均值，然后计算每个差值再平方。但这里直接调用均方差函数var()更加方便。
因为这里需要返回的是总方差，所以要用均方差乘以数据集中样本的个数 
'''
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0] #var() * 样本的个数

#将数据集格式化成目标变量Y和自变量X
def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    #X的第一列是偏差 Y是label
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0: #矩阵的逆不存在也会造成程序异常
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    #方程的正规解,最小二乘法
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

#生成叶子节点，之前的叶子节点是label的均值，构建数据的线性模型，叶子节点保存线性模型的权重
def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

#可以在给定的数据集上计算误差。与regErr() 类似， 
#会被 chooseBestSplit() 调用来找到最佳的切分。 该函数在数据集上调用 linearSolve()，
#之后返回yHat和Y之间的平方误差。
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2)) #返回yHat和Y之间的平方误差

'''
该函数还要确定什么时候停止切分，一旦停止切分会生成一个叶节点。用最佳方式切分数据集和生成相应的叶节点
leafType 是对创建叶节点的函数的引用， errType 是对 前面介绍的总方差计算函数的引用，  
ops 是一个用户定义的参数构成的元组， 用以完成树的构建。
它遍历所有的特征及其可能的取值来找到使误差最小化的切分阈值。
'''
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    #ops用户指定的参数，用于控制函数的停止时机。其中变量tolS是容许的误差下降值，tolN是切分的最少样本数
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    #统计不同剩余特征值的数目（label的取值个数）。如果该数目为1，那么就不需要再切分而直接返回
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet) #leafType(dataSet)是返回数据集label的平局值
    m,n = shape(dataSet) #计算当前数据集的大小
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet) #当前数据集的误差，用于与新切分误差进行对比，来检查新切分能否降低误差

    #下面对所有可能的特征及其 可能取值上遍历，找到最佳的切分方式。最佳切分也就是使得切分后能达到最低误差的切分。
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1): #对每个特征：
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]): #对每个特征值 .T转置
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)#将数据集切分成两份
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1) #新的误差等于两步分的误差和
            if newS < bestS:  #如果当前误差小于当前最小误差
                bestIndex = featIndex #那么将当前切分的属性设定为最佳切分
                bestValue = splitVal  #记录当前属性的取值
                bestS = newS          #更新最小误差
    
    if (S - bestS) < tolS:  #如果误差减少程度小于阈值（ops的第一个元素,那么就不应进行切分操作而直接创建叶节点
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue) 
    #检查两个切分后的子集大小，如果某个子集的大小小于用户定义的参数tolN，那么也不应切分
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue #返回切分的特征 以及对应切分的值
'''
参数：数据集和其他3个可选参数。leafType给出建立叶节点的函数；errType代表误差计算函数； 
ops是一个包含树构建所需其他参数的元组。
如果满足停止条件， chooseBestSplit()将返回None和某类模型的值 。
如果构建的是回归树，该模型是一个常数。如果是模型树，其模型是一个线性方程。
后面会看到停止条件的作用方式。
'''
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    #首先将数据集分成两个部分
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    # 如果不满足停止条件，会将创建一个新的Python字典并将数据集分成两份，
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #将数据集分成两份，
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    #在这两份数据集上将分别继续递归调用createTree()函数。
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

#测试输入变量是否是一棵树，用于判断当前处理的节点是否是叶节点
def isTree(obj):
    return (type(obj).__name__=='dict')

'''
函数getMean()是一个递归函数，它从上往下遍历树直到叶节点为止。如果找到两个叶节点则计算它们的平均值。
该函数对树进行剪枝处理（即返回子树平均值，用来作为新的叶子节点），在prune()函数中调用该
'''
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

#剪枝处理，参数待剪枝的树与剪枝所需的测试数据    
def prune(tree, testData):
    #函数首先需要确认测试集是否为空
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree 剪枝整个树
    #一旦非空，则反复递归调用函数 prune()对测试数据进行切分
    #要检查某个分支到底是子树还是节点。如果是子树，就调用函数prune()来对该子树 进行剪枝
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #如果两个分支 都是叶子，那么看看能否进行合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #没有合并的误差
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2)) 
        #合并叶子，取平均值
        treeMean = (tree['left']+tree['right'])/2.0
        #合并叶子后的误差
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean #返回一个叶子
        else: return tree #返回原来的树
    else: return tree

#普通的回归树，叶子节点是一个值  
def regTreeEval(model, inDat):
    return float(model)

#模型树到达叶子节点后，用线性模型进行预测
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

'''
预测数据;自顶向下遍历整棵树，直到命中叶节点为止,一旦到达叶节点，
它就会在输入数据上调用modelEval()函数，而该函数的默认值是regTreeEval()
参数modelEval：是对叶节点数据进行预测的函数的引用，包括普通的回归树regTreeEval, 和模型树modelTreeEval
'''
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']: #数据的属性大于阈值，跳到左子树
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval) #左子树不为空，递归调用
        else: return modelEval(tree['left'], inData) #叶子节点,预测
    else:  #数据的属性小于阈值，跳到右子树
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)

#对整个测试集进行预测, 循环调用 treeForeCast     
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat