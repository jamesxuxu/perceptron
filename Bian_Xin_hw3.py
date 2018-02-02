#!/usr/bin/python3

######################
## HW3 by Xin Bian  ##
######################

import numpy as np
import csv
#import matplotlib.pyplot as plt
import argparse

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
                    markLoc = int(element.split(':')[0])
                    temp[markLoc] = 1
            
            x.append(temp)
    return x, y
    
#function used to train model

eta = 0.1
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


#read data
xTrain, yTrain = readData('train')
xTest, yTest = readData('test')
xDev, yDev = readData('dev')

##test result on test dataset and dev set

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', action='store', dest='nstep', type=int, default=1)
    parser.add_argument('--capacity', action='store', dest='cap', type=int, default=0.868)
    argu = parser.parse_args()
    
    nstep = argu.nstep
    C = argu.cap

        
    w = np.zeros((124,1))
    for i in range(nstep):
        w = model(w)
        if i == nstep-1:
            ##test on test
            accTe = acc(xTest, yTest)
            ##test on train
            accTr = acc(xTrain, yTrain)
            ##test on dev
            accDe = acc(xDev, yDev)

    
    print('EPOCHS:', nstep)  
    print('CAPACITY:', C) 
    print('TEST_ACCURACY:', accTe)
    print('TRAINING_ACCURACY:', accTr) 
    print('DEV_ACCURACY:', accDe) 
    print('FINAL_SVM:', ' '.join(map(str, np.transpose(w)[0])))
