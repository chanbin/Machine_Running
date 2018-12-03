from Active_Function import Sigmoid, Softmax
from numpy import matrix, dot, empty, reshape

def BackPropDelta(Learning_rate, Active_Function, Bias, Weight, X, D, count_Training_Data, count_Layer, count_Node):
	alpha = Learning_rate

	for k in range(0,count_Training_Data):
		x = reshape(matrix(X[k]),(1,25)).transpose()
		d = matrix(D[k]).transpose()

		v = list()
		y = list()
		e = list()
		delta = list()
		for layer in range(count_Layer):
			v.append(None)
			e.append(None)
			delta.append(None)
			y.append(None)
			tmp = list()

			if layer == 0:
				v[layer] = dot(matrix(Weight[layer]), x) + Bias[layer] # 0은 bias
				for data in v[layer]:
					tmp.append([Sigmoid(data[0], False)])
			elif layer == (count_Layer-1):
				v[layer] = dot(matrix(Weight[layer]), matrix(y[layer-1])) + Bias[layer] # 0은 bias
				#tmp.append([Softmax(v[layer])])
				tmp = Softmax(v[layer])
			else:
				v[layer] = dot(matrix(Weight[layer]), matrix(y[layer-1])) + Bias[layer] # 0은 bias
				for data in v[layer]:
					tmp.append([Sigmoid(data[0], False)])
		
			y[layer] = tmp

		# Backpropagation
		for layer in range(count_Layer-1, 0-1, -1):
			if layer == (count_Layer-1):
				e[layer] = matrix(d) - matrix(y[layer])
			else:
				e[layer] = dot(matrix(Weight[layer+1]).transpose(), matrix(delta[layer+1]))
			
			delta[layer] = matrix(empty((count_Node[layer],1)))
			for i in range(0,count_Node[layer]):
				delta[layer][i] = y[layer][i][0] * (1-y[layer][i][0]) * e[layer][i][0]

		# Delta(Weight Mod)
		dW = list()
		for layer in range(count_Layer):
			dW.append(None)
			if layer == 0:
				dW[layer] = dot(alpha*matrix(delta[layer]), matrix(x).transpose()).tolist()
			else:
				dW[layer] = dot(alpha*matrix(delta[layer]), matrix(y[layer-1]).transpose()).tolist()
			
			Weight[layer] = Weight[layer] + dW[layer]

def BackPropMmt(Learning_rate, Active_Function, Bias, Weight, X, D, count_Training_Data, count_Layer, count_Node):
	alpha = Learning_rate

	#set momentum
	mmt = list()
	beta = 0.9
	for layer in range(count_Layer):
		mmt.append([[float(0) for _ in range(len(Weight[layer][0]))] for _ in range(len(Weight[layer]))])

	#train
	for k in range(0,count_Training_Data):
		x = reshape(matrix(X[k]),(1,25)).transpose()
		#x = matrix([sum(X[k],[])]).transpose()
		#tmp_x = list()
		#for i in range(len(X[k])):
		#	for j in range(len(X[k][0])):
		#		tmp_x.append(X[k][i][j])
		#x = matrix([tmp_x]).transpose()
		d = matrix(D[k]).transpose()
		v = list()
		y = list()
		e = list()
		delta = list()
		for layer in range(count_Layer):
			v.append(None)
			e.append(None)
			delta.append(None)
			y.append(None)
			tmp = list()

			if layer == 0:
				v[layer] = (dot(matrix(Weight[layer]), x) + Bias[layer]).tolist() # 0은 bias
				for data in v[layer]:
					tmp.append([Sigmoid(data[0], False)])
			elif layer == (count_Layer-1):
				v[layer] = (dot(matrix(Weight[layer]), matrix(y[layer-1])) + Bias[layer]).tolist() # 0은 bias
				tmp = Softmax(v[layer]).tolist()
			else:
				v[layer] = (dot(matrix(Weight[layer]), matrix(y[layer-1])) + Bias[layer]).tolist() # 0은 bias
				for data in v[layer]:
					tmp.append([Sigmoid(data[0], False)])
		
			y[layer] = tmp

		# Backpropagation
		for layer in range(count_Layer-1, 0-1, -1):
			if layer == (count_Layer-1):
				e[layer] = matrix(d) - matrix(y[layer])

			else:
				e[layer] = dot(matrix(Weight[layer+1]).transpose(), matrix(delta[layer+1]))
			
			delta[layer] = matrix(empty((count_Node[layer],1)))
			for i in range(0,count_Node[layer]):
				delta[layer][i] = y[layer][i][0] * (1-y[layer][i][0]) * e[layer][i][0]

		# Delta(Weight Mod)
		dW = list()
		for layer in range(count_Layer):
			dW.append(None)
			if layer == 0:
				dW[layer] = dot(alpha*delta[layer], matrix(x).transpose()).tolist()
			else:
				dW[layer] = dot(alpha*delta[layer], matrix(y[layer-1]).transpose()).tolist()

			mmt[layer] = dW[layer] + (beta * matrix(mmt[layer]))
			Weight[layer] = Weight[layer] + mmt[layer]