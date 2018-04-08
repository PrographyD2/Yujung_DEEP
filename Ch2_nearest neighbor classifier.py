import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
    #remember the training data
    #학습단계 모든 training데이터 메모리 올려서 기억 -> assign
    #X : 이미지, y : 해당 이미지 레이블
    self.Xtr = X
    self,ytr = y

    def predict(self, X):  #X : test 이미지
        num_test = X.Shape[0]

        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in xrange(num_test):

            distances = np.sum(np.abs(self.X[i,:]), axis = 1)
            #test이미지 한장을 5만개의 학습이미지와 비교 -> L1 distance 계산
            #L1 distance 기준으로 해당 test이미지와 가장 가까운, L1작은 학습 이미지 찾기
            # -> test image 레이블 예측
            #training 이미지에 따른 분류 작업의 속도는? linearly하게 증가, 학습 데이터 2배 -> 분류 시간 2배
            #training 속도보다 test속도가 더 중요,

            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

            return Ypred