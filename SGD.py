import random
from Delta import Delta
from Active_Function import Sigmoid

count_Training_Data = 4
count_Node = 1

X = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
D = [0,0,1,1]
Weight = list()
for i in range(0,3):
	Weight.append(random.uniform(-1,1))
bias = [0]
Learning_rate = 0.9

for epoch in range(0, 10000):
	Delta("SGD", Learning_rate, Sigmoid, bias, Weight, X, D, count_Training_Data, count_Node)

for k in range(0,count_Training_Data):
	v = 0
	for W_count in range(0,len(Weight)):
		x = X[k][W_count]
		v = v + (Weight[W_count]*x + 0)
	print(Sigmoid(v, False))