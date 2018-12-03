import math
#from sympy import symbols, Derivative

def Sigmoid(x, deff=False):#활성함수
	if deff:
		return sigmoid(x)*(1-sigmoid(x)) # sigmoid derivative
	else:
		return 1 / (1 + math.exp(-x)) # sigmoid func, set default

def Delta(Learning_rate, Active_Function, Bias, Weight, X, D, count_Training, count_Node):
	alpha = Learning_rate #학습률(0<alpha<=1), 크면 수렴하지 못함, 작으면 정답에 근접속도가 느림
				#W는 노드의 가중치(행렬), 입력데이터의 갯수만큼 리스트
	for k in range(0,count_Training):
		for node in range(0,count_Node):
			# 1~4까지 노드 하나
			# 1. 가중합
			v = 0
			for W_count in range(0,len(Weight)):
				x = X[k][W_count] #학습데이터의 입력데이터
				#v = W*x+0 #노드의 가중합(가중행렬*입력데이터+bias)
				#의도가 담긴 방향성
				v = v + (Weight[W_count]*x + Bias[node])
			d = D[k] #학습데이터의 정답데이터

			# 최종v = v + 0(행렬계산 + bias)
			# 2. 값 계산
			y = Active_Function(v, False) #신경망의 출력

			# 3. Delta규칙에 따른 출력오차 계산
			e = d - y # 출력오차(델타값) = 정답데이터 - 출력		
			#발전된 델타규칙: 함수(sigmoid)의 도함수y값 * 출력오차(e)
			delta = y*(1-y)*e

			# 4. 가중치 갱신
			dW = list()
			for W_count in range(0,len(Weight)):
				x = X[k][W_count]
				dW.append(alpha * delta * x) #가중치 갱신값
				Weight[W_count] = Weight[W_count] + dW[W_count]
			#첫 가중치는 랜덤값임
			#결국은 입력한 정답으로 유도하도록 가중치를 학습시킴
			#학습된 가중합으로 유추한 정답을 도출

#도함수 구하기 예시
#x = symbols('x')
#fx = 3*x**2 -4*x + 1
#f;x = Derivative(fx, x).doit()
#y; = f;x.subs({x:2})