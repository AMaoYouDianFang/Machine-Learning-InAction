'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *

#创建了一个用于测试的简单数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
'''
函数createC1()将构建集合C1。C1是只有一个元素的所有候选项集的集合。Apriori 算法首先构建集合C1，
然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。那些满足最低要求的项集构成集合L1。
而L1中的元素相互组合构成C2，C2再进一步过滤变为L2。
#C1是一个集合的集合，如{{0},{1},{2},…}，每次添加的都是单个项构成的集合{0}、{1}、{2}…
'''
def createC1(dataSet):
    C1 = [] #它用来存储所有不重复的项值
    for transaction in dataSet: 
        for item in transaction: 
            if not [item] in C1: #如果某个物品项没有在C1中出现，则将其添加到 C1中
                C1.append([item]) #添加的是一个列表
                
    C1.sort()
    '''
    frozenset是指被“冰冻”的集合，就是说它们是不可改变的，即用户不能修改它们。
    这里必须要使用frozenset而不是set类型，因为之后必须要将这些集合作为字典键值使用，
    使用frozenset可以实现这一点，而set却做不到，
    frozenset是冻结的集合，它是不可变的，存在哈希值，好处是它可以作为字典的key，
    也可以作为其它集合的元素。缺点是一旦创建便不能更改，没有set中的add，remove方法
    '''
    return list(map(frozenset, C1))#use frozen set so we
                            #can use it as a key in a dict    

#求频繁项集列表
#输入分别是数据集、候选项集列表Ck 以及感兴趣项集的最小支持度minSupport。用于从C1生成L1
#返回一个 包含支持度值的字典以备后用
def scanD(D, Ck, minSupport):
    ssCnt = {} #存储Ck中的所有候选集 对应出现的次数
    for tid in D: #遍历数据集中的所有交易记录
        for can in Ck:  #Ck中的所有候选集。 例如ck = c1
            if can.issubset(tid): #如果C1中的集合是记录的一部分，增加字典中对应的计数值
                if can not in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D)) 
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems #计算候选集的支持度
        if support >= minSupport:
            retList.insert(0,key) #retList 满足最小支持度要求的集合，存储支持度大于某个阈值的元素
            #在列表的首部插入任意新的集合。当然也不一定非要在首部插入，这只是为了让列表看起来有组织。
        supportData[key] = support #存储每个元素的支持度
    return retList, supportData

#创建候选项集Ck，函数aprioriGen()的输入参数为频繁项集列表Lk与项集元素个数k，输出为Ck
#举例来说， 该函数以{0}、{1}、{2}作为输入，会生成{0,1}、{0,2}以及{1,2}。 此时 Lk = {0}、{1}、{2} k = 2
#k表示要生成的集合，每个元素的长度，比如生成C2，这里k = 2
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk) #计算Lk中的元素数目
    #比较Lk中的每一个元素与其他元素，通过两个for循环来实现
    for i in range(lenLk): 
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2] #解释见下
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList
'''
取列表中的两个集合进行比较。如果这两个集合的前面k-2个元 素都相等，那么就将这两个集合合成一个大小为k的集合 。
这里使用集合的并操作来完成，在 Python中对应操作符|。
上面的k-2有点让人疑惑。接下来再进一步讨论细节。当利用{0}、{1}、{2}构建{0,1}、{0,2}、 {1,2}时，
这实际上是将单个项组合到一块。现在如果想利用{0,1}、 {0,2}、 {1,2}来创建三元素项集，应该怎么做？
如果将每两个集合合并，就会得到{0, 1, 2}、 {0, 1, 2}、 {0, 1, 2}。也就是说， 
同样的结果集合会重复3次。接下来需要扫描三元素项集列表来得到非重复结果，我们要做的是确保遍历列表的次数最少。
现在，如果比较集合{0,1}、 {0,2}、 {1,2}的第1个元素并只对第1个元素相同的集合求并操作，又会得到什么结果？
{0, 1, 2}，而且只有一次操作！这样就不需要遍历列表来寻找非重复值。
'''

#函数传递一个数据集以及一个支持度，函数会生成候选项集的列表
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet) #首先创建C1然后读入数据集将其转化为D（集合列表）
    D = list(map(set, dataSet)) #使用map函数将set()映射到dataSet列表中的每一项。
    L1, supportData = scanD(D, C1, minSupport) #scanD()函数来创建L1
    L = [L1] #将L1放入列表L中。L会包含L1、L2、L3…。
    k = 2
    #继续找L2，L3…，直到下一 个大的项集为空。
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k) #创建Ck
        # Ck是一个候选项集列表，然后scanD()会遍历Ck，丢掉不满足最小支持度要求的项集
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk #基于Ck来创建Lk。
        supportData.update(supK) 
        L.append(Lk) #Lk列表被添加到L，同时增加k的 值
        k += 1
    return L, supportData

'''
关联规则生成函数
函数generateRules()是主函数，它调用其他两个函数。 其他两个函数是rulesFromConseq()和calcConf()，
分别用于生成候选规则集合以及对规则进行评估。

有3个参数：频繁项集列表、包含那些频繁项集支持数据的字典(去掉了支持度低的组合)、最小可信度阈值。
函数最后要生成一个包含可信度的规则列表，后面可以基于可信度对它们进行排序。这些规则存放在bigRuleList中。
'''
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    '''
    该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1。因为无法从单元素项集中构建关联规则，
    所以要从包含两个或者更多元素的项集开始规则构建过程。如果从集合{0,1,2}开始，那么H1应该是[{0},{1},{2}]。
    如果频繁项集的元素数目超过2，那么会考虑对它做进一步的合并。
    '''
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet] #{0,1,2} --> [{0},{1},{2}]
            if (i > 1):  
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else: #项集中只有两个元素，那么使用函数calcConf()来计算可信度值
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList 

#计算规则的可信度以及找到满足最小可信度要求的规则。函数会返回一个满足最小可信度要求的规则列表
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return 创建一个空列表prunedH保存这些规则
    for conseq in H: #遍历H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

'''
为从最初的项集中生成更多的关联规则，可以使用rulesFromConseq()函数。
该函数有2个参数：一个是频繁项集，另一个是可以出现在规则右部的元素列表H。
。接下来查看该频繁项集是否大到可以移除大小为m的子集。如果可以的话，则将其移 除。可以使用程序清单11-2中的函数aprioriGen()来生成H中元素的无重复组合 。该结果会 存储在Hmp1中，这也是下一次迭代的H列表。Hmp1包含所有可能的规则。可以利用calcConf() 来测试它们的可信度以确定规则是否满足要求。如果不止一条规则满足要求，那么使用Hmp1迭
'''
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0]) #函数先计算H中的频繁集 大小m 
    #可以利用calcConf() 来测试它们的可信度以确定规则是否满足要求。如果不止一条规则满足要求，那么使用Hmp1迭
    if (len(freqSet) > (m + 1)): #try further merging 
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates #生成H中元素的无重复组合,该结果会 存储在Hmp1中，这也是下一次迭代的H列表。Hmp1包含所有可能的规则
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf) #可以利用calcConf() 来测试它们的可信度以确定规则是否满足要求。如果不止一条规则满足要求，那么使用Hmp1迭
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f" % ruleTup[2])
        print()       #print a blank line
        
            
# from time import sleep
# from votesmart import votesmart
# votesmart.apikey = 'get your api key first'
# def getActionIds():
#     actionIdList = []; billTitleList = []
#     fr = open('recent20bills.txt') 
#     for line in fr.readlines():
#         billNum = int(line.split('\t')[0])
#         try:
#             billDetail = votesmart.votes.getBill(billNum) #api call
#             for action in billDetail.actions:
#                 if action.level == 'House' and \
#                 (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
#                     actionId = int(action.actionId)
#                     print('bill: %d has actionId: %d' % (billNum, actionId))
#                     actionIdList.append(actionId)
#                     billTitleList.append(line.strip().split('\t')[1])
#         except:
#             print("problem getting bill %d" % billNum)
#         sleep(1)                                      #delay to be polite
#     return actionIdList, billTitleList
        
# def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
#     itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
#     for billTitle in billTitleList:#fill up itemMeaning list
#         itemMeaning.append('%s -- Nay' % billTitle)
#         itemMeaning.append('%s -- Yea' % billTitle)
#     transDict = {}#list of items in each transaction (politician) 
#     voteCount = 2
#     for actionId in actionIdList:
#         sleep(3)
#         print('getting votes for actionId: %d' % actionId)
#         try:
#             voteList = votesmart.votes.getBillActionVotes(actionId)
#             for vote in voteList:
#                 if not transDict.has_key(vote.candidateName): 
#                     transDict[vote.candidateName] = []
#                     if vote.officeParties == 'Democratic':
#                         transDict[vote.candidateName].append(1)
#                     elif vote.officeParties == 'Republican':
#                         transDict[vote.candidateName].append(0)
#                 if vote.action == 'Nay':
#                     transDict[vote.candidateName].append(voteCount)
#                 elif vote.action == 'Yea':
#                     transDict[vote.candidateName].append(voteCount + 1)
#         except: 
#             print("problem getting actionId: %d" % actionId)
#         voteCount += 2
#     return transDict, itemMeaning
