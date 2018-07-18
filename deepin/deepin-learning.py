#!/usr/bin/python
#coding=utf-8

import numpy as np

def sigmoid(x,flag=False):
    if(flag):
        return x*(1-x)
    return 1/(1+np.exp(-x)) #exp完全是抄袭了，丝毫不懂
handle = open('UCI.txt')
lines = handle.readlines()
rows = len(lines)

x = np.zeros((rows,30))
y = np.zeros((rows,1))

#读取部分
row = 0
for line in lines:
    line =line.split('\t')
    x[row,:] = line[1:]
    if(line[0] == 'M'):
        y[row]=1
    else:
        y[row]=0
    row += 1

print(x)
print(x.shape)
print(y)
print(y.shape)

np.random.seed(1)#随机种子？不是很清楚

syn0 = 2*np.random.random((30,1)) -1

for z in range(100):
    for i in range(300):
        print(i)
        l0 = x[i].reshape(1,30)
        #print("al",l0,l0.shape)
        l1 = sigmoid(np.dot(l0,syn0))
        l1_error = y[i] - l1
        l1_delta =l1_error * sigmoid(l1,True)#误差加权平均值
        #print(l0.shape,l1_delta.shape)
        syn0 += np.dot(l0.T,l1_delta)


print("finish training")
print(syn0)

test = open("UCI_test.txt")
lines_test = test.readlines()
rows_test = len(lines_test)

x_test = np.zeros((rows,30))
y_test = np.zeros((rows,1))

row = 0
M_sum = 0
B_sum = 0
correct_sum = 0
for line in lines_test:
    line = line.split('\t')
    x[row,:] = line[1:]
    print(np.dot(x[row].reshape(1,30),syn0))
    if(line[0] == 'M'):
        M_sum += 1
        if(abs(np.dot(x[row].reshape(1,30),syn0)-1) <= 0.01):
            correct_sum += 1
    else:
        B_sum += 1
        if(abs(np.dot(x[row].reshape(1,30),syn0)) <= 0.01):
            correct_sum += 1
    row += 1

print("成功率",correct_sum/row)
