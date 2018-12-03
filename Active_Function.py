import math
from numpy import exp, max, sum
from random import sample

def Sigmoid(x, deff=False):
	if deff:
		return Sigmoid(x)*(1-Sigmoid(x)) #도함수
	else:
		return 1 / (1 + math.exp(-x))

def Softmax(array):
	ex = exp(array - max(array)) # 사랑스러운 overflow 방지용
	return ex / float(sum(ex, axis=0))

def ReLU(x, deff=False):
	if deff:
		if x > 0:
			return 1
		elif x <= 0:
			return 0
	else:
		if x > 0:
			return x
		elif x <= 0:
			return 0.01*x

def Dropout(y, ratio):
	row = len(y)
	col = len(y[0])
	drop_index  = sample(range(0,row), int(row*ratio))
	for idx, val in enumerate(y):
		if idx in drop_index:
			y[idx][0] = 0.0
		else:
			y[idx][0] = y[idx][0] * (1/(1-ratio))
	return y