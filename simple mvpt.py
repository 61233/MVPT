# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 20:06:24 2022

@author: HUAWEI

in the simple situation, the return rate of securities are uncorrelated,
i.e. the corvariance matrix is diagonal.
the theroy can be explained as a non-linear regression problem with riskaversion coefficient q = 1/2:
    r = [r1,...,rn]'  s = diag{s1,...,sn}
    x = alpha + 1/2 beta 
      = (2-sum{ri/si})/(sum{2/si}) [1/s1,...1/sn]' + 1/2 [r1,...,rn]'
    i.e.
    xk = (2-sum{ri/si})/(sum{2/si}) * 1/sk + 1/2 * rk/sk
the multiple-layer neural network is as follows

"""



'''
1. import the package
'''

import torch
import torch.nn as nn
import numpy as np
from torch import optim

import torch.utils.data 

from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt

'''
2. build the neural network
'''
# model 中数据是否需要拉平？
class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()
        self.l1 = nn.Sequential(nn.Linear(8,20),nn.BatchNorm1d(20),nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(20, 10),nn.BatchNorm1d(10),nn.ReLU())
        self.l3 = nn.Linear(10, 4)
    
    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

'''
3. create the data
input: mean and covariance of securities (generate) r,s
output: proportion of investment into security (calculate) x
dim = 4
inputdim = 8, outputdim = 4
'''
dim = 4
num_sample = 1000
r = 2*(torch.rand(num_sample,dim)-0.5)
s = torch.rand(num_sample,dim)
e = torch.ones(1,dim)
x = torch.zeros(num_sample,dim)
for i in range(num_sample):
    rc = r[i,:] # example of one sample
    sc = s[i,:] # example of one sample
    a = torch.sum(rc/sc); b = torch.sum(e/sc); c = (2-a)/(2*b); sce = e/sc; scr = rc/sc;
    xc = c*sce + 0.5*scr
    #print('rc',rc.numpy());print('sc',sc.numpy());print('xc',xc.numpy());print('test',torch.sum(xc).numpy())
    x[i,:] = xc
inputdata = torch.cat((r,s),dim = 1)
outputdata = x
#print('r',r.numpy());print('s',s.numpy());print('x',x.numpy())

'''
4. data process
dataset and dataloader
'''
if torch.cuda.is_available():
    inputdata = inputdata.cuda()
    outputdata = outputdata.cuda()

train_data, test_data, train_label, test_label = train_test_split(
    inputdata,outputdata,test_size=0.25,random_state=0)

train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
test_dataset = torch.utils.data.TensorDataset(test_data, test_label)

batch_size = 100 

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle = False)

'''
5. import the neural network
'''
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLPmodel().to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(),lr=learning_rate)

'''
6. train the model
'''

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    ct = 0 # count, to calculate average loss
    loss_total = 0 # calculate three batches' loss (enumerate%x=0)
    for batch, (X, y) in enumerate(dataloader): 
        # len(X) = 50 = batch_size
        # batch form 1 to 15 ,size(train_dataloader)=15
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  #梯度清零
        loss.backward()    #计算梯度
        optimizer.step()   #模型参数优化
        
        
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            ct += 1 #count, in this data situation(num=1000,train=750,ebatch%5) ct is 3, batch=0,5,10
            loss_total += loss  # calculate three batches' loss (enumerate%x=0)
            print(f"loss: {loss:>4f}  [{current:>5d}/{size:>5d}]")
    avg_loss = loss_total/max(ct,1)  # average_loss of each epoch
    lossall.append(avg_loss)  # record the avg_loss
    print(f"avg_loss:{avg_loss:>4f}") 


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset) # to calculate the average correct
    num_batches = len(dataloader) # to calculate the average test_loss; ??这里不懂size和num_batches有什么区别
    test_loss, correct = 0, 0 #iteration initial

    with torch.no_grad(): # do not execute the gradient calculation
        for X, y in dataloader:
            pred = model(X)  #prediction
            test_loss += loss_fn(pred, y).item() #calculate the test loss (on the whole testdataset)
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()  #raw program 
            #correct += (pred == y).type(torch.float).sum().item()  #??不知道pred.argmax(1)==y是什么意思，原问题为分类问题
            correct += torch.tensor((test_loss < 1e-2)).float().sum().item() #calculate the test loss

    test_loss /= num_batches #average testloss of each epoch
    correct /= size   #average correct of each epoch
    accuracy.append(correct)  #record
    testlossall.append(test_loss)  #record the average testloss
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

            
epochs = 500  # epoch times
lossall = []
accuracy = []
testlossall = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, criterion, optimizer)
    test_loop(test_dataloader, model, criterion)
    
    
plt.figure()
plt.plot(lossall)
plt.title('train loss, simple data')

plt.figure()
plt.plot(accuracy)
plt.title('test accuracy, simple data')

plt.figure()
plt.plot(testlossall)
plt.title('test loss, simple data')
print("Done!")


# testloss曾经不收敛但是改了改参数不知道怎么的就可能出现收敛，或者下降但是不收敛到0的情况
# 未进行数据的标准化，归一化和正则化

    

