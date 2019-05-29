'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from numpy import *
from time import sleep

#打开文件并对 其进行逐行解析，从而得到每行的类标签和整个数据矩阵
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
'''
随机选择alpha的下标
i是第一个alpha的下标，m是所有alpha的数目
'''
def selectJrand(i,m):
    j=i #只要函数值不等于输入值i，函数就会进行随机选择。
    while (j==i):
        j = int(random.uniform(0,m))
    return j

#用于调整大于H或小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj
'''
简化的SMO
参数：数据集、类别标签、常数C、容错率和退出前最大的循环次数
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    #转置了类别标签 得到的就是一个列向量而不是列表，类别标签向量的每行元素都和数据矩阵中的行一一对应
    labelMat = mat(classLabels).transpose() 
    b = 0; m,n = shape(dataMatrix)
    #构建一个alpha列矩阵，矩阵中元素都初始化为0，m个与元素对应m个alpha
    alphas = mat(zeros((m,1)))
    #iter用于记录没有任何alpha改变的情况下遍历数据集的次数
    #当该变量达到输入值maxIter时，函数结束运行并退出
    iter = 0
    while (iter < maxIter):
        #用于记录alpha是否已经进行优化，初始化为0
        alphaPairsChanged = 0
        for i in range(m): #对于每个alphas[i]
            #计算这个样本的预测值 multiply是对应元素相乘，由于dataMatrix是矩阵mat，这里*就是点乘
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            #基于这个实例的预测结果和真实结果的比对，就可以计算误差Ei
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            
            #判断样本是否满足KKT条件，解释见统计学习p129，if是不满足KTT的样本
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m) #随机选择第二个alpha值
                #计算第j样本对应的误差
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                #Python 则会通过引用的方式传递所有列表，所以必须明确地告知Python要为alphaIold和
                # alphaJold 分配新的内存；否则的话，在对新值和旧值进行比较时，我们就看不到新旧值的变化。
                alphaIold = alphas[i].copy() 
                alphaJold = alphas[j].copy()
                #计算L和H, 解释见jupyter
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                
                #如果L和H相等，就不做任何改变，直 接执行continue语句,本次循环结束直接运行下一次for的循环
                if L==H: print("L==H"); continue
                # Eta是alpha[j]的最优修改量 2K12 - K11 + K22
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                #如果eta为0，那么 计算新的alpha[j]比较麻烦,这里做了简化，不处理这种情况
                if eta >= 0: print("eta>=0"); continue
                #统计学习p127，计算更新后的值
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                #利用L与H值进行调整
                alphas[j] = clipAlpha(alphas[j],H,L)
                #如果更新后的alphas[j]与没有更新的相差比较小，放弃本次的计算
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                #利用alphas[j]更新alphas[i]，若y1y2相同，y1+ y2是特定的数，alphas[i]要加上 alphaJold - alphas[j]
                #y1y2不同，则相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                #更新b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                #标记成功更新一次
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print("iteration number: %d" % iter)
    return b,alphas

#==============================下面是完整的SVM （包括核函数）=========================================================
'''
核转换函数
参数: 2个数值 型变量和1个元组。
元组kTup给出的是核函数的信息元组的第一个参数是描述所用核函数类型的一个字符串
其他2个参数则都是核函数可能需要的可选参数
'''
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
        #在NumPy矩阵中，除法符号(/)意味着对矩阵元素展开计算而不像在MATLAB中一样计算矩阵的逆
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

'''
使用对象的目的并不是为了面向对象的编程，而只是作为一个数据结构来使用对象
除了增加了一个m×2的矩阵成 员变量eCache之外 ，这些做法和简化版SMO一模一样。
eCache的第一列给出的是eCache是 否有效的标志位，而第二列给出的是实际的E值(真实值-当前的预测值)。
'''
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        #矩阵K先被构建，然后再通过调用函数kernelTrans()进行填充。全局的K值只需计算一次
        #然后，当想要使用核函数时，就可以对它进行调用。这也省去了很多冗 余的计算开销。
        self.K = mat(zeros((self.m,self.m))) 
        for i in range(self.m): #填充K
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

#计算E值并返回 , Ek 预测值-真实值 
#。以前，该过程是采用内嵌 的方式来完成的，但是由于该过程在
# 这个版本的SMO算法中出现频繁，这里必须要将其单独拎出来     
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b) #changed for kernel
    Ek = fXk - float(oS.labelMat[k]) 
    return Ek
'''
选择第二个alpha或者说内循环的alpha值  
这里的目标是选择合适的第二个alpha值以保证在每次优化中采用最大步长
该函数的误差值与第一个 alpha值Ei和下标i有关。首先将输入值Ei在缓存中设置成为有效的。
这里的有效（valid）意味着它已经计算好了
'''      
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    #设置eCache 第i个的标记位为1，并记录Ei， 标记位为0说明还没有算到
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    #.A 意味着将数据类型从矩阵(mat)更改为数组(numpy)
    validEcacheList = nonzero(oS.eCache[:,0].A)[0] #返回的是非零E值所对应的alpha值，而不是E值本身
    if (len(validEcacheList)) > 1: #第2次循环之后
        #在所有的值上进行循环并选择其中使得改变（两个E差的绝对值）最大的那个值
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k) 
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej #maxK是选出的alpha_j
    #如果这是第一次循环的话，那么就随机选择一个alpha值
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

#alpha值进行优化后,计算误差值并存入缓存当中
def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

#寻找决策边界的优化例程       
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #使用新的方法选择alpha_j this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache #更新alpha_j的E值
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache   #更新alpha_j的E值          #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]  #changed for kernel
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]   #changed for kernel
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0
'''
外循环代码
其输入和函数smoSimple()完全一样。函数 一开始构建一个数据结构来容纳所有的数据，然后需要对控制函数退出的一些变量进行初始化。
整个代码的主体是while循环，这与smoSimple()有些类似，但是这里的循环退出条件更多一些。当迭代次数超过指定的最大值，
或者遍历整个集合都未对任意alpha对进行修改时，就退出循环

一开始是对全部的数据集进行扫描，然后进行非边界扫描（会跳过那些已知的不会改变的alpha值，节省时间）一直到非边界扫描
没有跟新，再进入全部扫描，一直到最后没有更新
'''
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0 #迭代次数，每次迭代有多个更新
    entireSet = True
    alphaPairsChanged = 0 #更新次数
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all #一开始的for循环在数据集上遍历任意可能的alpha
            for i in range(oS.m):    #遍历所有的样本，在innerL判断是否满足KKT    
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        #非边界扫描 会跳过那些已知的不会改变的alpha值
        else:#go over non-bound (railed) alphas 环遍历所有的非边界alpha值，也就是不在边界0或C上的值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

#利用alpha 计算w 虽然for循环遍历了数据集中的所有数据，
#但是最终起作用的只有支持向量 alpha>0。
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

#测试中使用核函数, 径向基函数有一个用户定义的输入
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    #运行Platt SMO算法
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    #找出那些非零的 alpha值，从而得到所需要的支持向量的id
    svInd=nonzero(alphas.A>0)[0]
    #获取支持向量
    sVs=datMat[svInd] #get matrix of only support vectors
    #获取支持向量的标签
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        #用结构初始化方法中使用过的kernelTrans()函数，得到转换后的数据
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        #预测结果，alpha及类别标签值求积
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))

    #下面只使用测试集
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m): 
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))    
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m)) 


########***********************************************************
# Non-Kernel VErsions below   没有Kernel函数的版本
########***********************************************************

class optStructK:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJK(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEkK(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEkK(oS, j)
    return j, Ej

def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEkK(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerLK(i, oS):
    Ei = calcEkK(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJK(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEkK(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEkK(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoPK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStructK(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0 
    alphaPairsChanged = 0 #更新次数
    entireSet = True
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all 
            for i in range(oS.m):   # 遍历所有的样本 在innerLK 判断样本是否满足KKT 
                alphaPairsChanged += innerLK(i,oS) #更新的次数+1
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLK(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas