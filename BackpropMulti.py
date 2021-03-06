import random
from Backpropagation_Multiclass import BackPropDelta, BackPropMmt
from Active_Function import Sigmoid, Softmax
from numpy import matrix, dot, reshape

X = [[[0,1,1,0,0],	#1
	  [0,0,1,0,0],
	  [0,0,1,0,0],
	  [0,0,1,0,0],
	  [0,1,1,1,0]],

	 [[1,1,1,1,0],	#2
	  [0,0,0,0,1],
	  [0,1,1,1,0],
	  [1,0,0,0,0],
	  [1,1,1,1,1]],

	 [[1,1,1,1,0],	#3
	  [0,0,0,0,1],
	  [0,1,1,1,0],
	  [0,0,0,0,1],
	  [1,1,1,1,0]],

	 [[0,0,0,1,0],	#4
	  [0,0,1,1,0],
	  [0,1,0,1,0],
	  [1,1,1,1,1],
	  [0,0,0,1,0]],

	 [[1,1,1,1,1],	#5
	  [1,0,0,0,0],
	  [1,1,1,1,0],
	  [0,0,0,0,1],
	  [1,1,1,1,0]]]

D = [[1,0,0,0,0],
	 [0,1,0,0,0],
	 [0,0,1,0,0],
	 [0,0,0,1,0],
	 [0,0,0,0,1]]

count_Training_Data = len(X)

Node = [25,50,5] #[입력,은닉,출력]

Weight = list()
Bias = list()
for i in range(0,len(Node)-1):
	Weight.append([[float(0) for _ in range(Node[i])] for _ in range(Node[i+1])])
	Bias.append([[float(0)] for _ in range(Node[i+1])])

count_Layer = len(Weight)

count_Node = list()
for layer in range(0, count_Layer):
	count_Node.append(len(Weight[layer]))
count_Node.append(1)

for layer in range(0, count_Layer):
	for node in range(0, count_Node[layer]):
		for index in range(0, len(Weight[layer][node])):
			Weight[layer][node][index] = random.uniform(-1,1)

Learning_rate = 0.9

for epoch in range(0, 300):
	if epoch%100==0:
		print("epoch: "+str(epoch))
	#BackPropDelta(Learning_rate, Sigmoid, Bias, Weight, X, D, count_Training_Data, count_Layer, count_Node)
	BackPropMmt(Learning_rate, Sigmoid, Bias, Weight, X, D, count_Training_Data, count_Layer, count_Node)
print("")

for k in range(0,count_Training_Data):
	x = reshape(matrix(X[k]),(1,25)).transpose()
	v = list()
	y = list()
	for layer in range(count_Layer):
		v.append(None)
		y.append(None)
		tmp = list()

		if layer == 0:
			v[layer] = dot(matrix(Weight[layer]), x) + Bias[layer] # 0은 bias
			for data in v[layer]:
				tmp.append([Sigmoid(data[0], False)])
		elif layer == (count_Layer-1):
			v[layer] = dot(matrix(Weight[layer]), matrix(y[layer-1])) + Bias[layer] # 0은 bias
			#tmp.append(Softmax(v[layer]))
			tmp = Softmax(v[layer]).tolist()
		else:
			v[layer] = dot(matrix(Weight[layer]), matrix(y[layer-1])) + Bias[layer] # 0은 bias
			for data in v[layer]:
				tmp.append([Sigmoid(data[0], False)])
		
		y[layer] = tmp
	
	for i in range(len(y[1])):
		for j in range(len(y[1][0])):
			#print(round(y[1][i][j],15))
			print('{:.15f}'.format(y[1][i][j]))
	print("")