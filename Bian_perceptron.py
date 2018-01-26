#!/usr/bin/python3

######################
## HW2 by Xin Bian  ##
######################

import numpy as np
import csv
#import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', action='store', dest='nstep', type=int, default=1)
parser.add_argument('--nodev', action='store_true', dest='noDev', default=False)
argu = parser.parse_args()

nstep = argu.nstep
noDev = argu.noDev

#define read data function
def readData(dataset):
    y = []
    x = []
    path = ['/u/cs246/data/adult/a7a' , dataset]
    filepath = '.'.join(path)
    with open(filepath) as f:
        trainData = csv.reader(f, delimiter=' ')
        for row in trainData:
            y.append(int(row[0]))
    
            temp = np.zeros((124,1), dtype=np.int)  
            temp[-1] = 1
            for element in row[1:]:
                if len(element) != 0:
  		    #things are confusing here
                    #the feature index can be start from 1 or 0.
                    #if considering 1 to 123, I should use markLoc = int(element.split(':')[0]) - 1
                    #to be consistent with the example, I consider it 0 to 122: 
                    markLoc = int(element.split(':')[0])
                    temp[markLoc] = 1
            
            x.append(temp)
    return x, y
    
#read data
xTrain, yTrain = readData('train')
xTest, yTest = readData('test')

#function used to train model
def model(nstep):
    w = np.zeros((124,1))
    for _ in range(nstep):
        for index, xn in enumerate(xTrain):
            t = np.sign(np.dot(np.transpose(w), xn))        
            if t!= yTrain[index]:
                w = w + yTrain[index]*xn 
    return w
    

    
##test result on test dataset
right = 0
w = model(nstep)
for index, xn in enumerate(xTest):
    t = np.sign(np.dot(np.transpose(w), xn))
    if t*yTest[index] > 0:
        right += 1
accuracy = right/float(len(xTest))

    
print('Test accuracy:', accuracy)
print('Feature weights (bias last):', ' '.join(map(str, np.transpose(w)[0])))
    

##dev to adjust hyperparameter
acc = []
if noDev == False:
    xDev, yDev = readData('dev')
    for i in range(nstep):
        right = 0
        wDev = model(i+1)
        for index, xn in enumerate(xDev):
            t = np.sign(np.dot(np.transpose(wDev), xn))
            if t*yDev[index] > 0:
                right += 1
        acc.append(right/float(len(xDev)))
    print('dev accuracy vs. iteration:', acc)
   # plt.plot(acc)
   # plt.show()
    
    
    
            
