'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

#导入马疝病数据集
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    #计算特征数目，包括最后的label
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr) 
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#函数将用于测试是否有某个值小于或者大于我们正在测试的阈值 通过阈值比较对数据进行分类的。 
# 所有在阈值一边的数据会分到类别-1，而在另外一边的数据分到类别+1
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1)) #将返回数组的全部元素设置为1
    if threshIneq == 'lt': #所有不满足不等式要求的元素设置为1
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    
#遍历stumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树
def buildStump(dataArr,classLabels,D, isPrint = True):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    #用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf #minError则在一开始 就初始化成正无穷大，之后用于寻找可能的最小错误率
    for i in range(n):# 遍历所有的特征
        #计算当前特征的取值范围（特征是数值类型）
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        #计算步长，遍历numSteps步
        stepSize = (rangeMax-rangeMin)/numSteps
        #在当前的特征的值上遍历-1 -numSteps 共numSteps+ 2次
        #将阈值设置为整个取值范围之外也是可以的。因此，在取值范围之外还应该有两个额外的步骤
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #有两种可能，比阈值小的预测为1，比阈值大的预测为1
                #分界值，第一次取比rangeMin小的数
                threshVal = (rangeMin + float(j) * stepSize)
                #返回分类预测结果，只根据某一个属性的值预测
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                #误差矩阵，分类错误的是1
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                #将错误向量errArr和权重向量D的相应元素相乘并求和，就得到了误差率weightedError
                weightedError = D.T*errArr  #calc total error multiplied by D
                if isPrint: print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:  #更新最小的误差率
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    # 在词 典bestStump中保存该单层决策树
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

'''
函数名称尾部的DS代表的就是单层决策树（decision stump），它是AdaBoost中最流行的弱分类器，
当然并非唯一可用的弱分类器。上述函数确实是建立于单层决策树之上的，但是我们也可以很容易对此进行修改
以引入其他基分类器。实际上，任意分类器都可以作为基分类器，本书前面讲到的任何一个算法都行。
上述算法会输出一个单层决策树的数组，因此首先需要建立一个新的Python表来对其进行存储。
然后，得到数据集中的数据点的数目m，并建立一个列向量D。

样本的权重都赋予了相等的值。在后续的迭代中，AdaBoost算法会在增加错分数据的权重的同时，
降低正确分类数据的权重
'''
def adaBoostTrainDS(dataArr,classLabels,numIt=40, isPrint = True):
    weakClassArr = [] 
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #每个数据点的权重 初始化为1/m 保证和为1
    aggClassEst = mat(zeros((m,1))) #记录每个数据点的类别估计累计值。
    for i in range(numIt): #循环运行numIt次或者直到训练错误率为0为止
        # buildStump()函数建立一个单层决策树
        bestStump,error,classEst = buildStump(dataArr,classLabels,D, isPrint)#build Stump
        if isPrint: print("D:",D.T)
        #当前分类器的权重值 alpha = 1/2 * ln( (1- error) / error) error是单层决策树计算的加权分类误差
        alpha = float(0.5*log((1.0-error)/max(error,1e-16))) # max(error, 1e-16)用于确保在没有错误时不会发生除零溢出
        bestStump['alpha'] = alpha   # alpha值加入到 bestStump字典中
        weakClassArr.append(bestStump)  #该字典又添加到列表中。该字典包括了分类所需要的所有信息
        if isPrint: print("classEst: ",classEst.T)
        #计算下一次迭代中的新权重向量D classLabels 真实的label classEst预测的label
        # 若预测和真实的一样，相乘是1，即只有 -1*alpha 不一样相乘是-1，alpha
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) 
        D = multiply(D,exp(expon))                             
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        #计算累计的估计值 分类器的权重 * 预测结果
        aggClassEst += alpha*classEst
        if isPrint: print("aggClassEst: ",aggClassEst.T)
        #计算累加分类错误，如果错误aggErrors对应的位置是1
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        if isPrint: print("total error: ",errorRate)
        #训练错误率为0时，就要提前 结束for循环
        if errorRate == 0.0: break
    return weakClassArr

# 对数据进行预测，利用训练出的多个弱分类器进行分类 的函数
def adaClassify(datToClass,classifierArr):
    #首先将待分类样例datToClass转换成了一个NumPy矩阵
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0] 
    aggClassEst = mat(zeros((m,1))) #与 adaBoostTrainDS()中的含义一样，存储分类器累加的预测结果。
    for i in range(len(classifierArr)): #多个弱分类器组成的数组，遍历classifierArr中的所有弱分类器
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        #输出的类别估计值乘上该单层决策树的alpha权重然后累加到 aggClassEst 上
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print(aggClassEst[:10])
    return sign(aggClassEst)

#ROC曲线的绘制及AUC计算函数
# predStrengths 未使用sigmod的预测值， classLabels 真实的label
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) # 元组保留的是绘制光标的位置
    ySum = 0.0 # 用于计算AUC的变量
    numPosClas = sum(array(classLabels)==1.0) #计算正样本的数量
    yStep = 1/float(numPosClas) #y坐标轴的步长
    xStep = 1/float(len(classLabels)-numPosClas) #len(classLabels)-numPosClas 负样本的数量， x的步长
    #计算排序索引，索引是按照最小到最大的顺序排列的
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #要从点<1.0,1.0>开始绘，一直到<0,0>
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]: #Python需要一个表来进行迭代循 环，因此我们需要调用tolist()方法
        if classLabels[index] == 1.0: #当遍历表时，每得到一个标签为1.0的类，则要沿着y轴的方向下降一个步长，即不断降低真阳率
            delX = 0
            delY = yStep
        else:  #在x轴方向上倒退了一个步长（假阴率方向）
            delX = xStep
            delY = 0
            ySum += cur[1]  #高度的累加
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b') #画出点
        cur = (cur[0]-delX,cur[1]-delY) #更新点
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)

    '''
    为了计算AUC，我们需要对多个小矩形的面积进行累加。这些小矩形的宽度是xStep，
    因此 可以先对所有矩形的高度进行累加，最后再乘以xStep得到其总面积。
    所有高度的和（ySum）随着x轴的每次移动而渐次增加。一旦决定了是在x轴还是y轴方向上进行移动的，
    我们就可以在当 前点和新点之间画出一条线段。然后，更新当前点cur。最后，我们就会得到一个像样的绘图并 
    将AUC打印到终端输出。
    '''
