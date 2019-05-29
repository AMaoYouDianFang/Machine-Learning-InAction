'''
Created on Oct 19, 2010

@author: Peter
'''
import numpy as np

'''
创建了一些实验样本 该函数返回的第一个变量是进行词条切分后的文档集合，
这些文档来自斑点犬爱好者留言板。这些留言文本被切分成一系列的词条集合， 
标点符号从文本中去掉，后面会探讨文本处理的细节。
'''
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    return postingList, classVec

#朴素贝叶斯词集模型
#创建一个包含在所有文档中出现的不重复词的列表,每一个单词构建一个特征
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        #将每篇文档返回的新词集合添加到该集合中,操作符|用于求 两个集合的并集，这也是一个按位或（OR）操作符
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)
#输出文档向量，向量的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    #遍历文档中的所 有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#输入参数为文档矩阵trainMatrix，以及由标签向量 trainCategory
#返回值 p0Vect : p(w i |c =0 )  p1Vect : p(w i |c =1 )  pAbusive p(c =1 )
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  #文档的数目（文档是一句话）
    numWords = len(trainMatrix[0]) #每个文档单词属性的个数
    #计算文档属于侮辱性文档（class=1）的概率，即P(1)，可以通过1-P(1)得到P(0)
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #初始化算p(w i |c 1 ) 和p(w i |c 0 )
    #p0Num 是在标签是0的情况下，每个单词出现的数目
    p0Num = np.ones(numWords) 
    #p1Num 是在标签是1的情况下，每个单词出现的数目
    p1Num = np.ones(numWords)      #change to np.ones()
    # 记录标签是1样本的所有单子的总数
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):#遍历文档
        #类别是1的时候一旦某个词语出现，则该词对应的个数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #p(w i |c 1 ) 是 p1Num/p1Denom 每个单词出现的次数/ 该类别所有单词的总数
    p1Vect = np.log(p1Num/p1Denom)          #change to np.log()
    p0Vect = np.log(p0Num/p0Denom)          #change to np.log()
    return p0Vect, p1Vect, pAbusive

# 这里有疑问？？？
# 分类 输入要分类的向量vec2Classify以及使用函数trainNB0()计算得到的三个概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #对应元素 相乘，即先将两个向量中的第1个元素相乘，然后将第2个元素相乘，以此类推。
    #接下来将词汇表 中所有词的对应值相加，然后将该值加到类别的对数概率上
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#朴素贝叶斯词袋模型，统计了每个单词出现的次数，前面只是记录单词是否出现
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

#接受一个大字符串并将其解析为字符串列表。
#该函数去掉少于两 个字符的字符串，并将所有字符串转换为小写
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
'''
第二个函数spamTest()对贝叶斯垃圾邮件分类器进行自动化处理。
导入文件夹spam与ham 下的文本文件，并将它们解析为词列表
'''
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    #构建一个测试集与一个训练集 其中的10封电子邮件被随机选择 为测试集
    trainingSet = range(50); testSet = []           #create test set
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        #其从训练集中剔除
        del(list(trainingSet)[randIndex])
    trainMat = []; trainClasses = []
    #for循环遍历训练集的所有文档，对每封邮件基于词汇表并使用setOfWords2Vec() 函数来构建词向量
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    #预测
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))
    #return vocabList, fullText

def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet = []           #create test set
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
