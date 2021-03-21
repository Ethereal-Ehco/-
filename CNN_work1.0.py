import numpy as np
import torch
import sklearn.preprocessing as sp
import os
import random
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torchvision import transforms

#构建数据集
root='D:\\毕业设计\\优化\\data\\数据集3.0'
def MyLoader(path):
    dataset=[]
    label_dic={'A':0,'N':1,'L':2,'R':3,'P':4,'V':5}
    with open(path,'r') as f:
        tmp=f.readlines()
        dataset.append(label_dic[tmp[0].rstrip('\n')])
        for eachdata in tmp[1:]:
            data=eachdata.rstrip('\n')
            dataset.append(float(data))
    return dataset


class MyDataset(Dataset):
    # 构造函数设置默认参数
    def __init__(self, path):
        datasets=[[],[]]
        dataMax=0.0
        dataMin=1000
        txt_path= os.listdir(path)
        random.shuffle(txt_path)
        for i in range(len(txt_path)):
            loadpath=path+'\\'+txt_path[i]
            data=MyLoader(loadpath)
            '''
            if max(data[1:3])>dataMax:
                dataMax=max(data[1:3])
            if min(data[1:3])<dataMin:
                dataMin=min(data[1:3])
            '''
            datasets[0].append([data[1:]])
            '''
            #print(datasets)
            datasets[0][i][0][0]=(datasets[0][i][0][0]-dataMin)/(dataMax-dataMin)
            datasets[0][i][0][1]=(datasets[0][i][0][1]-dataMin)/(dataMax-dataMin)
            '''
            datasets[1].append(data[0])
        self.x=torch.Tensor(datasets[0])
        self.y=torch.Tensor(datasets[1])
        self.txt_path=txt_path
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):

        return len(self.txt_path)



class CNN_Net(torch.nn.Module):
    def __init__(self):
        super(CNN_Net,self).__init__()
        self.conv1=torch.nn.Conv1d(1,5,kernel_size=(62,))
        self.conv2=torch.nn.Conv1d(5,5,kernel_size=(62,))
        self.pooling=torch.nn.MaxPool1d(2)
        self.liner1=torch.nn.Linear(58*5,128)
        self.liner2=torch.nn.Linear(128,64)
        self.liner3=torch.nn.Linear(64,6)
    def forward(self,x):
        #print(x.shape)
        x=F.relu(self.conv1(x))
        #print('conv1',x.shape)
        x=self.pooling(x)
        #print('pooling1',x.shape)
        x=F.relu(self.conv2(x))
        #print('conv2',x.shape)
        #x=self.pooling(x)
        #print('pooling2',x.shape)
        x=x.view(-1,58*5)
        x=F.relu(self.liner1(x))
        x=F.relu(self.liner2(x))
        return self.liner3(x)

model=CNN_Net()
criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

train_data = MyDataset(path=root+'\\'+'trainingData' )
test_data = MyDataset(path=root + '\\' + 'testData')
#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
trainloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
testloader = DataLoader(dataset=test_data, batch_size=32)



def train(epoc):
    running_loss=0.0
    for batch_idx,data in enumerate(trainloader):
        #print(batch_idx)
        inputs,target=data
        target=torch.Tensor(target).long()
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if batch_idx%200==199:
            print('[%d,%5d] loss: %.3f' % (epoc + 1, batch_idx + 1, running_loss / 200))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            testData, labels = data
            outputs = model(testData)
            _, predict = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    print('Accuary on test set:%.2f %%' % (100 * correct / total))

if __name__=='__main__':
    for epoc in range(100):
        train(epoc)
        test()




