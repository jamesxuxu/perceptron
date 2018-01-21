######################
## HW2 by Xin Bian  ##
######################

import numpy as np
import csv
import matplotlib.pyplot as plt

y = []
x = []


def readData(dataset):
    path = ['./adult/a7a' , dataset]
    filepath = '.'.join(path)
    with open(filepath) as f:
        trainData = csv.reader(f, delimiter=' ')
        for row in trainData:
            y.append(int(row[0]))
    
            temp = np.zeros((124,1), dtype=np.int)  
            temp[-1] = 1
            for element in row[1:]:
                if len(element) != 0:
                    markLoc = int(element.split(':')[0]) - 1
                    temp[markLoc] = 1
            
            x.append(temp)
    
    
readData('train')

#train model, update w
def model(alpha):
    w = np.zeros((124,1))
    for index, xn in enumerate(x):
        t = np.sign(np.dot(np.transpose(w), xn))        
        if t!= y[index]:
            w = w + alpha*y[index]*xn 
    return w



readData('train')
right = 0
acc = []

#change learning rate, plot accuracy
for alpha in range(1,101,2):
    alpha = alpha/float(100)
    w = model(alpha)
    for index, xn in enumerate(x):
        t = np.sign(np.dot(np.transpose(w), xn))
        if t*y[index] > 0:
            right += 1
    acc.append(right/float(len(x)))
    right = 0

plt.plot(acc)
    
    
    
            