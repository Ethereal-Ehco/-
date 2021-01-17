from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
import os
import random
import collections

def getData(path):
    labels=[]# 分类标签
    dataLists=os.listdir(path)
    random.shuffle(dataLists)
    m=len(dataLists)#数据集的数量
    dataSets=np.zeros((m,301))#数据集
    i=0#行
    for dataList in dataLists:
        f=open(path+'\\'+dataList,'r')
        tmp=f.readlines()
        f.close()
        labels.append(tmp[-1])
        tmp=tmp[0:-1]
        j=0#列
        for each in tmp:
            n=each.rstrip('\n')
            n=float(n)
            dataSets[i][j]=n
            j+=1
        i+=1
    dataSets=np.mat(dataSets)
    return dataSets,labels

def Classifier(dataSets,labels,testsDataSets,testLabels):
    neigh=KNN(n_neighbors=2)
    neigh.fit(dataSets,labels)
    accuary=0
    mtest=len(testLabels)
    labelsNum=collections.Counter(testLabels)
    result={}
    for each in labelsNum.keys():
        result[each]=0
    print(result)
    for i in range(mtest):
        testMat=testsDataSets[i]
        classifierResult=neigh.predict(testMat)
       # print(classifierResult)
        if classifierResult==testLabels[i]:
            result[testLabels[i]]+=1
            accuary+=1
    accuary=accuary/mtest
    for each in result.keys():
        result[each]/=labelsNum[each]
    print(result)
    print('正确率为：%.2f%%'%(accuary*100))
if __name__=='__main__':
    path1='D:\\毕业设计\\测试\\数据集\\trainingData'
    path2='D:\\毕业设计\\测试\\数据集\\testData'
    path3='D:\\毕业设计\\测试\\数据集\\convxData'
    dataSets,labels=getData(path1)
    testDataSets,testLabels=getData(path2)
    extrayData,extrayLabel=getData(path3)
    Classifier(np.vstack((dataSets,extrayData)),labels+extrayLabel,testDataSets,testLabels)



