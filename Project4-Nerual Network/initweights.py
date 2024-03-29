# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:32:45 2019

@author: Jerry Xing
"""
import numpy as np
import random 

def initweights(wst):
#    % function W=initweights(wst);
#    % 
#    % returns a randomly initialized weight vector for a given neural network
#    % architecture.
	#random.seed(717)
	entry = np.cumsum(wst[0:-1] * wst[1:] + wst[0:-1]) # entry points into weights
	W = np.random.randn(entry[-1], 1) / 2
	
	return W



# print(initweights(wst=np.array([1,20,20,20,13])).shape)