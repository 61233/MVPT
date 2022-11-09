# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:35:23 2022

@author: HUAWEI

input: mean and covariance data of return rate, r and s
output: the proportion of investment , x

for the unrestricted dim-asset portfolio, x is as follows:
    risk averse coefficient: q = 1/2 in the program 
    x(t) = alpha + q*beta 
    a = e'*(s^{-1})*e >0
    b = e'*(s^{-1})*r
    c = r'*(s^{-1})*r
    d = ac-b^2 >0
    alpha = 1/a*(s^{-1})*e
    beta = (s^{-1})*(r-b/a*e)
    x = x(1/2) = alpha + 1/2*beta
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

class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()
        self.l1 = nn.Sequential(nn.Linear(20,100),nn.BatchNorm1d(100),nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(100, 50),nn.BatchNorm1d(50),nn.ReLU())
        self.l3 = nn.Linear(50, 4)
    
    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

'''
3. create the true data for model: 
input: mean and covariance data of return rate, r and s
output: the proportion of investment , x
   
'''

def x_calculation(dim,r,s,t=1/2):
    '''
    parameters:
    dim: int 
    r: 1*dim vector/ 2dtensor with shape 1*dim
       mean matrix/vector
    s: dim*dim matrix/ 2dtensor with shape dim*dim 
       covariance matrix
    t: int. the defaullt is 1/2
    return:
    x: 1*dim vector/ 2dtensor with shape 1*dim
        
    # test x_calculation
    s = torch.tensor([[1,0,0],[0,2,0],[0,0,3]])
    r = torch.tensor([[1.],[2.],[0.]])
    r = r.T
    x = x_calculation(3,r,s)
    #succeed
    '''
    s = s.float()
    s1 = torch.inverse(s)
    e = torch.ones(1,dim)
    a = torch.mm(torch.mm(e,s1),e.T) #torch.Size([1,1])
    b = torch.mm(torch.mm(e,s1),r.T) #torch.Size([1,1])
    #c = torch.mm(torch.mm(r,s1),r.T) #torch.Size([1,1])
    #d = torch.mm(a,c)-torch.mm(b,b) #torch.Size([1,1]) 
    alpha = 1/a*torch.mm(e,s1)
    #print('alpha size',alpha.size()) #torch.Size([1,dim])
    #print('alpha',alpha.numpy())
    rd = r - b/a*e  #torch.Size([1,dim])
    beta = torch.mm(rd,s1)  #torch.Size([1,dim])
    x = alpha + torch.mul(t,beta) #torch.Size([1,dim])
    #print('test',torch.sum(x).numpy())  # the correct answer is 1

    return x

def total_x_calculation(data_finance,num_sample,dim):
    x = torch.ones(num_sample,dim)
    for i in range(num_sample):
        r = data_finance[i,0:dim]  # the (i+1)th sample r
        r = r.reshape(1,-1)
        s = data_finance[i,dim:]  # the (i+1)th sample s
        s = s.reshape(dim,dim)
        #print(s.size())
        xc = x_calculation(dim,r,s)
        for j in range(dim):
            x[i,j] = xc[:,j]
    return x

def rs_creation(num_sample,dim):
    '''
    Parameters
    ----------
    num_sample : int
        the number of samples/ row number of the sample matrix
    dim : int
        the number of dimension/ relate to column number of the sample matrix

    Returns
    -------
    matrix with shape: num_sample*(dim+dim*dim)
    
    # test rs_creation
    data_finance = rs_creation(50,3)
    s = data_finance[:,3:]
    #print('s',s.numpy())
    print('s size',s.size())
    shang = s[0,:]
    shang = shang.reshape(3,3)
    eig = torch.eig(shang)
    print('one s',shang.numpy())
    print('eig value',eig)
    #print('data_finance',data_finance.numpy())
    print('one data_finance',data_finance[0,:])
    print('data_finance size',data_finance.size())
    '''
    data_finance = torch.ones(num_sample,dim+dim*dim)
    for i in range(num_sample):
        r = torch.rand(1,dim)
        s = torch.rand(dim,dim)
        s = torch.mm(s,s.T)
        hang = s.reshape(1,-1)
        hang = torch.cat((r,hang),dim=1)
        for j in range(dim+dim*dim):
            data_finance[i,j]=hang[:,j]
     
    return data_finance

num_sample = 2000 # the number of the sample
dim = 4  # the dimension of the question, dim implies the number of securities
inputdata = rs_creation(num_sample,dim)
outputdata = total_x_calculation(inputdata, num_sample, dim)
#print('x',outputdata.numpy())
#print('x size',outputdata.size())

# to plot the mu-sigma figure, we need to calculate the true mu,sigma and the model prediction mu,sigma

def musigma(data,x,dim,row):
    '''
    Parameters
    ----------
    data : 
        shape: row * dim+dim^2
        mean and covariance of return rate
    x :
        shape: row * dim
        proportion of investment
    dim : int, related to column dimension
    row : int, row dimension

    Returns
    -------
    mu: mean of the portfolio, mu = <x,r>
    sigma2: variance of the portfolio, sigma2 = <x,sx> 
    注:这里的算式x代表列向量，程序数据中均为行向量形式

    '''
    mu = torch.zeros(row,1)  # size: row*1
    sigma2 = torch.zeros(row,1)
    r = data[:,0:dim]
    #print('r',r.size())
    s = data[:,dim+1:]
    #print('s',s.size())
    for i in range(row):  #program question:len(s)??
        xk = x[i,:].unsqueeze(0)
        #print(x.size())
        rk = r[i,:].unsqueeze(0)
        #print(rk.size())        
        sk = s[i,:]
        sk = sk.reshape(dim,dim)
        #print(sk.size())
        muk = torch.sum(rk*xk)
        sigma2k = torch.sum(xk*torch.mm(xk,sk))
        #print(sigma2.size())
        sigma2[i,:] = sigma2k
        mu[i,:] = muk
        
    return mu,sigma2


#%%
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

batch_size = 50 

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
    return pred,y,X
   


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset) # to calculate the average correct
    num_batches = len(dataloader) # to calculate the average test_loss; 
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
    return pred,y,X

            
epochs = 500  # epoch times
lossall = []
accuracy = []
testlossall = []
#true_mu = []; true_sigma2 = [];
#cal_mu = []; cal_sigma2 = [];
#data = inputdata[0,:].unsqueeze(0)
#label = outputdata[0,:].unsqueeze(0) # to calculate the mu and sigma in iteration
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    pred,y,d = train_loop(train_dataloader, model, criterion, optimizer)
    #sp_pred = model(data)
    #mut,sigma2t = musigma(data, label, dim, 1)
    #muc,sigma2c = musigma(data,sp_pred,dim,1)
    #true_mu.append(mut);true_sigma2.append(sigma2t);cal_mu.append(muc);cal_sigma2.append(sigma2c)
    predt,yt,dt = test_loop(test_dataloader, model, criterion)


#print('pred train',pred.size())
#print('y train', y.size())
#print('d train', d.size())
#print('predt test', predt.size())
#print('yt test', yt.size())
#print('dt test',dt.size())

#print(len(true_mu))

#%%
# plot one of the model prediction and true label in the train loop
plt.figure()
pred1 = pred[0,:]
y1 = y[0,:]
#pred1 = pred1.cpu().detach().numpy()
#y1 = y1.cpu().detach().numpy()
print('pred1',pred1)
print('y1',y1)
plt.plot(pred1,y1,'g')


# plot mu-sigma line
# 暂时未画出，曲线是由风险厌恶系数为参变量，代码中规定
# 要画出的点是固定某一组输入输出，画出随着迭代的进行由不同的x得到不同的mu,sigma2
# 如果模型收敛，最终画出来的序列点将会收敛到理想musigma2曲线上的某个固定点，其他信息暂时不会提供





# plot train_loss-epoch
plt.figure()
plt.plot(lossall)
plt.title('train loss, complex data')
print(min(torch.tensor(lossall)))

# plot test_loss-epoch
plt.figure()
plt.plot(accuracy)
plt.title('test accuracy, complex data')

plt.figure()
plt.plot(testlossall)
plt.title('test loss, complex data')
print("Done!")


# done! nn.BatchNorm1d()  未进行数据的标准化
# 未进行数据的正则化，归一化
# 数据是否统一flatten 还没做系统的考察（参考图像分类问题的模板代码）