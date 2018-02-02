#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 22:26:33 2018

@author: Xin
"""
import numpy as np
from matplotlib import pyplot
import Bian_Xin_hw3




testAcc = []
devAcc = []

Cpool = np.arange(-3,4.1,0.1)
Cpool = np.power(10, Cpool)
for C in Cpool:
    w = np.zeros((124,1))
    for _ in range(5):
        w = model(w)
    
    testAcc.append(acc(xTest, yTest))
    devAcc.append(acc(xDev, yDev))
    
pyplot.plot(Cpool, testAcc, color='blue',label='Test dataset')
pyplot.plot(Cpool, devAcc, color='black',label='Dev dataset')
pyplot.xscale('log')
pyplot.xlabel('Capacity')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.savefig('plot.png')
pyplot.show()