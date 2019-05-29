'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *

#它将文本文件导入到一个列表中。文本文件每一行为tab分隔的浮点数
#返回值是一个包含许多其他列表的列表。这种格式可以很容易将很多值封装到矩阵中
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

#计算两个向量的欧式距离，也可以使用其他的距离函数
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)
'''
为给定数据集构建一个包含k个随机质心的集合
随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。
然后生成0到1.0之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内
'''
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ) #第j个属性最大值最小值的差值
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

#4个参数:只有数据集及簇的数目是必选参数，而用来计算距离和创建初始质心的函数都是可选的   
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] #确定数据集中数据点的总数
    #后创建一个矩阵来存储每个点的簇分配结果
    #簇分配结果矩阵clusterAssment 包含两列：一列记录簇索引值，第二列存储误差。
    #误差是指当前点到簇质心的距离,后边会使用该误差来评价聚类的效果。
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    #随机初始化k个点                                  
    centroids = createCent(dataSet, k)
    clusterChanged = True  #如果该值为True，则继续迭代
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k): #遍历每个质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2 #更新当前样本的簇头
        print(centroids)
        for cent in range(k):#recalculate centroids 
            #首先通过数组过滤来获得给定簇的所有点
            # clusterAssment[:,0] 所有样本对应的簇头索引
            #获取簇头是k的样本， .A 是把np.mat类型转换成np.array
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment 

#二分聚类
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    #创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = mat(zeros((m,2)))
    #计算整个数据集的质心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    print(centroid0)
    #并使用一个列表来保留所有的质心, 初始是centroid0，后面会继续加入
    centList =[centroid0] #create a list with one centroid
    #遍历数据集中所有 点来计算每个点到质心的误差值
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    #不停对簇进行划分，直到得到想要的簇数目为止
    while (len(centList) < k):
        lowestSSE = inf #一开始将最小SSE置设为无穷大
        for i in range(len(centList)): #遍历簇列表centList中的每 一个簇，选取划分后误差（sseSplit + sseNotSplit）最小的簇进行划分
            #将 该 簇 中 的 所 有 点 看 成 一 个 小 的 数 据 集 ptsInCurrCluster
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            #将当前簇进行二均值处理，会新生成两个簇，同时给出每个簇的误差值
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #计算属于簇i节点的数据划分后的误差之和
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            #计算其他簇（非i的簇）的误差和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            #划分的误差与剩余数据集的误差之和作为本次划分的误差，如果该划分的SSE值最小，则本次划分被保存
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i #记录当前划分的簇编号
                bestNewCents = centroidMat #新划分的两个簇头的数据
                bestClustAss = splitClustAss.copy() #新划分簇内的两个簇的误差
                lowestSSE = sseSplit + sseNotSplit #最好的误差
        #只需要将要划分的簇中所有点的簇分配结果进行修改即可，kMeans()函数并且指定簇数为2时，
        # 会得到两个编号分别为0和1的结果簇。需要将这些簇编号修 改为划分簇及新加簇的编号 
        # 编号为0的改为len(centList)  ，编号为1的改为原来划分的簇编号
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        #将centList中原先的族数据（不是编号）,更换为新划分后类别编号为0的簇头数据
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        #centList追加划分后类别编号为1的簇头数据
        centList.append(bestNewCents[1,:].tolist()[0])
        #更进行新簇划分点的误差值
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment #返回簇中心数据， 每个样本对应的簇编号，以及到簇头的误差
#每个簇编号，和对应的数据 （簇编号1 数据[1.1 2.1 3.1]）

import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    c=urllib.request.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
