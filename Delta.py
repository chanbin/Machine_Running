from Active_Function import Sigmoid
#from sympy import symbols, Derivative

def Delta(Weight_Update, Learning_rate, Active_Function, Bias, Weight, X, D, count_Training_Data, count_Node):
	alpha = Learning_rate
	if(Weight_Update == "BATCH"): dwSum = [0.0] * len(Weight)
	for k in range(0,count_Training_Data):
		for node in range(0,count_Node):
			v = 0
			for W_count in range(0,len(Weight)):
				x = X[k][W_count]
				v = v + (Weight[W_count]*x + Bias[node])
			d = D[k]

			y = Active_Function(v, False)

			e = d - y
			delta = y*(1-y)*e

			dW = list()
			for W_count in range(0, len(Weight)):
				x = X[k][W_count]
				dW.append(alpha * delta * x)
				if(Weight_Update == "SGD"):
					Weight[W_count] = Weight[W_count] + dW[W_count]
				elif(Weight_Update == "BATCH"):
					dwSum[W_count] = dwSum[W_count] + dW[W_count]
	
	if(Weight_Update == "SGD"):
		pass
	elif(Weight_Update == "BATCH"):
		dwAvg = [0.0] * len(Weight)
		for W_count in range(0, len(Weight)):
			dwAvg[W_count] = dwSum[W_count] / count_Training
			Weight[W_count] = Weight[W_count] + dwAvg[W_count]