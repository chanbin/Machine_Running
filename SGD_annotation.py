import random
from Delta import Delta, Sigmoid

#N = int(input("학습데이터의 수: "))
count_Training = 4 # 학습데이터의 수
count_Node = 1

# 1. 데이터 입력
X = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
D = [0,0,1,1]
Weight = list()
for i in range(0,3): # 노드의 입력값들의 가중치 count
	Weight.append(random.uniform(-1,1)) # 가중치 랜덤값으로 초기화
bias = [0]
Learning_rate = 0.9 #학습률(0<alpha<=1), 크면 수렴하지 못함, 작으면 정답에 근접속도가 느림

for epoch in range(0, 10000): # SGD 방식
	Delta(Learning_rate, Sigmoid, bias, Weight, X, D, count_Training, count_Node) #활성함수(시그모이드)를 델타 규칙으로 학습

for k in range(0,count_Training):
	v = 0
	for W_count in range(0,len(Weight)):
		x = X[k][W_count] #학습데이터의 입력데이터
		#v = W*x+0 #노드의 가중합(가중행렬*입력데이터+bias)
		#의도가 담긴 방향성
		v = v + (Weight[W_count]*x + 0)
	# 최종v = v + 0(행렬계산 + bias) 
	print(Sigmoid(v, False))