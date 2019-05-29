'''
Created on Jan 8, 2011

@author: Peter
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#用来计算最佳拟合直线
def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    #计算XTX，
    xTx = xMat.T*xMat
    #判断它的行列式是否为零，如果行列式为零，那么计算逆矩阵的时候将出现错误
    #linalg.det()来计算行列式
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    #NumPy的线性代数库还提供一个函数来解未知矩阵
    #ws=linalg.solve(xTx, xMat.T*yMatT)
    return ws

#局部加权线性回归函数
#给定x空间中的任意一点，计算出对应的预测值yHat
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    #创建对角权重矩阵 weights，阶数等于样本点个数，该矩阵为每个样本点初始化了一个权重
    weights = mat(eye((m))) 
    for j in range(m):   #算法将遍历数据集，计算每个样本点对应的权重值   #next 2 lines create weights matrix
        #随着样本点与待预测点距离的递增，权重将以指数级衰减
        #预算每个样本点与待预测点的差
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2)) #这里使用的是diffMat的平方，不是pdf的绝对值
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

#为数据集中每个点调用lwlr()
def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

#岭回归，用于计算回归系数，
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam  #单位矩阵eye
    if linalg.det(denom) == 0.0:  #如果lambda设定为0的时候一样可能会产生错误，所以这里仍需要做一个检查
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

# 预测
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    #对数据进行标准化
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30 #以在30个不同的lambda下调用ridgeRegres()函数
    wMat = zeros((numTestPts,shape(xMat)[1]))
    #这里的lambda 应以指数级变化，这样可以看出lambda在取非常小的值时和取非常大的值时分别对结果造成的影响。
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

#逐步线性回归算法的实现 lasso 算法。
#eps 每次迭代需要调整的步长， numIt，表示迭代次数
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #用于记录每次迭代后的结果
    ws = zeros((n,1))
    #建立了ws的两份副本
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        #print(ws.T) #每次迭代时都打印出w向量
        lowestError = inf; 
        for j in range(n): #遍历所有的特征
            for sign in [-1,1]: #分别增加，减小这个特征的取值
                wsTest = ws.copy()
                wsTest[j] += eps*sign  
                yTest = xMat*wsTest  ##更新特征后计算预测值
                rssE = rssError(yMat.A,yTest.A) #计算平方误差
                if rssE < lowestError: 
                    lowestError = rssE #更新最小误差
                    wsMax = wsTest     #更新最好的w
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

#def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print "item #%d did not sell" % i
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print "%s\t%d\t%s" % (priceStr,newFlag,title)
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()
    
from time import sleep
import json
import urllib.request
#调用Google购物API并保证数据抽取 的正确性
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10) #为了防止短时间内有过多的API调用
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read()) #打开和解析操作，完成后我们将得到一个字典
    #在这些产品上循环迭代，判断该产品是否是新产品并抽取它的价格
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                '''
                不完整的套装也会通过检索结果返回，所以我们需要将这些信息过滤掉
                （可以统计描述中的关键词或者是用贝叶斯方法来判 断）。这里仅使用了一个简单的启发式方法：
                如果一个套装的价格比原始价格低一半以上， 则认为该套装不完整。
                '''
                if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print('problem with item %d' % i)

#它负责多次调用searchForSet()来获取数据，参数是从www.brickset.com收集来的    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

#交叉验证测试岭回归 numVal是交叉验证的次数
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal,30))#create error mat 30columns numVal rows

    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m): #建立训练集测试集，将数据 集的90%分割成训练集，其余10%为测试集
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #ridgeTest()使用30个不同的λ值创建了30组不同的回归系数
        for k in range(30):#用30组回归系数来循环测试回归效果
            #岭回归需要使用标准化后的数 据
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            #预测结果，为什么加上mean(trainY) ？？？？
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
            errorMat[i,k]=rssError(yEst.T.A,array(testY)) #计算误差并且保存
            #在所有交叉验证完成后，errorMat保存了ridgeTest()里每个λ对应的多个误差值
            #print errorMat[i,k]

    '''
    在所有交叉验证完成后，errorMat保存了ridgeTest()里每个λ对应的多个误差值。
    为了 将得出的回归系数与standRegres()作对比，需要计算每个λ对应误差值的均值
    '''
    meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors)) #找到具有最小误差的对应的权重
    bestWeights = wMat[nonzero(meanErrors==minMean)]

    #岭回归使用了数据标准化
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    #不明白2 ？？？？
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))