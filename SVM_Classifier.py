from sklearn import svm
import numpy as np
import os
import random

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

def classifier(trainingDataSets,trainingLabels,testDataSets,testLabels):
    clf=svm.SVC(kernel='rbf')
    clf.fit(trainingDataSets,trainingLabels)
    result=clf.predict(testDataSets)
    accuary = 0
    mtest = len(testLabels)
    for i in range(len(result)):
        if result[i] == testLabels[i]:
            accuary += 1
    accuary = accuary / mtest
    print('正确率为：%.2f%%' % (accuary * 100))

if __name__=='__main__':
    path1='D:\\毕业设计\\测试\\数据集\\trainingData'
    path2='D:\\毕业设计\\测试\\数据集\\testData'
    trainingDataSets,trainingLabels=getData(path1)
    testDataSets,testLabels=getData(path2)
    classifier(trainingDataSets,trainingLabels,testDataSets,testLabels)