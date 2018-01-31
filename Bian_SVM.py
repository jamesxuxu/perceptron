#!/usr/bin/python3

######################
## HW2 by Xin Bian  ##
######################

import numpy as np
import csv
#import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', action='store', dest='nstep', type=int, default=1)
parser.add_argument('--nodev', action='store_true', dest='noDev', default=False)
argu = parser.parse_args()

nstep = argu.nstep
noDev = argu.noDev
#define read data function
def readData(dataset):
    y = []
    x = []
    path = ['./adult/a7a' , dataset]
    filepath = '.'.join(path)
    with open(filepath) as f:
        trainData = csv.reader(f, delimiter=' ')
        for row in trainData:
            y.append(int(row[0]))
    
            temp = np.zeros((124,1), dtype=np.int)  
            temp[0] = 1
            for element in row[1:]:
                if len(element) != 0:
                    #things are confusing here
                    #the feature index can start from 1 or 0.
                    #if considering 1 to 123, I should use markLoc = int(element.split(':')[0]) - 1
                    #to be consistent with the example, I consider it 0 to 122:
                    markLoc = int(element.split(':')[0])
                    temp[markLoc] = 1
            
            x.append(temp)
    return x, y
    
#read data
xTrain, yTrain = readData('train')
xTest, yTest = readData('test')

if noDev == False:
    xDev, yDev = readData('dev')

eta = 0.1
C = 0.868


#function used to train model
def model(wPre):
    w = wPre
    for index, xn in enumerate(xTrain):
        t = np.dot(np.transpose(w), xn)       
        if t*yTrain[index]>1:
            w[1:] -= eta*w[1:]/(float(len(xTrain)))
        else:
            w[1:] -= eta*(w[1:]/(float(len(xTrain))) - C*yTrain[index]*xn[1:])
            w[0]  +=  eta*C*yTrain[index]
    return w

##accuracy function
def acc(x, y):
    right = 0
    for index, xn in enumerate(x):
        t = np.sign(np.dot(np.transpose(w), xn))
        if t*y[index] > 0:
            right += 1
    accuracy = right/float(len(x))
    return accuracy

##test result on test dataset and dev set
accDev = []
w = np.zeros((124,1))
for i in range(nstep):
    w = model(w)
    if noDev == False:
        right = 0
        for index, xn in enumerate(xDev):
            t = np.sign(np.dot(np.transpose(w), xn))
            if t*yDev[index] > 0:
                right += 1
        accDev.append(right/float(len(xDev)))
    if i == nstep-1:
        ##test on test
        accTe = acc(xTest, yTest)
        ##test on train
        accTr = acc(xTrain, yTrain)

print('EPOCHS:', nstep)  
print('CAPACITY:', C) 
print('TEST_ACCURACY:', accTe)
print('TRAINING_ACCURACY:', accTr)  
print('DEV_ACCURACY:', accDev[0])
print('Feature weights (bias first):', ' '.join(map(str, np.transpose(w)[0])))
print(np.transpose(w)[0])
if noDev == False:
    print('dev set accuracy vs. iterations:', accDev)
