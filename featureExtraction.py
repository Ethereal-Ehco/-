import wfdb
import os
import shutil
import argparse
import random
import matplotlib.pyplot as plt
def FeatureExtra():
    fileName=[100,101,102,103,104,105,106,107,108,109,
              111,112,113,114,115,116,117,118,119,
              121,122,123,124,200,201,202,203,205,207,
              208,209,210,212,213,214,215,217,219,
              220,221,222,223,228,230,231,232,233,234]
    labeldic={'N':0,'R':0,'L':0,'V':0,'A':0}
    for dataNum in fileName:
        #待处理数据路径
        dataPath='D:\\毕业设计\\测试\\data\\processedData\\'+str(dataNum)+'.txt'
        fd=open(dataPath,'r')
        data=fd.readlines()
        fd.close()
        for i in range(len(data)):
            data[i]=data[i].rstrip('\n')
        #心电信号头文件路径
        headerPath = 'D:\\毕业设计\\测试\\data\\MIT-BIH\\' + str(dataNum)
        signal_annotation = wfdb.rdann(headerPath, "atr")
        #R波的位置
        position=signal_annotation.sample
        #R波的类型
        labels=signal_annotation.symbol
        for index,label in enumerate(labels):
        #index指第index个R波
        #label值R波的标签
            if label in labeldic.keys():
                labeldic[label]+=1
            #对每种类型的R波新建一个文件夹储存
                filePath = 'D:\\毕业设计\\测试\\dataSets\\'+label
                if not os.path.exists(filePath):
                    os.makedirs(filePath)
            #正确的存储方式为以N为例 N文件夹下，每一个txt存储301个数据，命名编号依此递增
                f=open(filePath+'\\'+str(dataNum)+'_'+str(labeldic[label])+'.txt','w')
                left=position[index]-150
                right=position[index]+151
                if left < 0:
                    left = 0
                if right > len(data):
                    right = len(data)
                for i in range(left,right):
                    f.write(data[i]+'\n')
                f.write(label)
                f.close()
            else:
                pass
        print('文件已处理完毕',dataNum)
    return labeldic
def chooseDataSets():
    firstdir='D:\\毕业设计\\测试\\dataSets'
    tardir='D:\\毕业设计\\测试\\数据集\\convxData'
    #获取文件夹下所有的文件夹
    pathdir=os.listdir(firstdir)
    #print(pathdir)
    #得到'N','L','R','A','V' 5个文件夹的路径
    #path：每个文件夹的路径
    filePath=[]
    for path in pathdir:
        filePath.append(firstdir+'\\'+path)
    #print(filePath)
    newName=0
    for i in filePath:
        sample=random.sample(os.listdir(i),400)
        #print(sample)
        for txtDoc in sample:
            shutil.move(i+'\\'+txtDoc,tardir+'\\'+txtDoc)
            os.rename(tardir+'\\'+txtDoc,tardir+'\\'+str(newName)+'.txt')
            newName+=1
        print(len(os.listdir(i)))
    print('done!')
def showDataSets():
    firstdir = 'D:\\毕业设计\\测试\\dataSets'
    pathdir = os.listdir(firstdir)
    filePath = []
    for path in pathdir:
        filePath.append(firstdir + '\\' + path)
    plt.figure()
    for i in filePath:
        sample=random.sample(os.listdir(i),1)
        for txtDoc in sample:
            f=open(i+'\\'+txtDoc,'r')
            data=f.readlines()
            data=data[:-1]
            for i in range(len(data)):
                data[i] = float(data[i].rstrip('\n'))
            print(data)
            plt.plot(data)
    plt.show()




if __name__=='__main__':
    #labelInfo=FeatureExtra()
    #print(labelInfo)
    chooseDataSets()
    #showDataSets()


