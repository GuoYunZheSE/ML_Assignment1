import numpy as np
import sklearn
import matplotlib.pyplot as plt
import time
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


class ALS:


    def __init__(self,R,Parameters):
        self.Rating_Matrix=R
        self.rows=Parameters['Row']
        self.cols=Parameters['Cols']
        self.K=Parameters['K']
        self.m_lambda=Parameters['Lambda']
        self.max_loops=Parameters['Max Loops']
        self.epsilon=Parameters['epsilon']


    def iteraton(self, matrix_fixed, fixed_vecs):
        tic=time.time()
        if matrix_fixed== 'P_Matrix':
            num_solve=self.cols
        else:
            num_solve=self.rows
        num_fixed=fixed_vecs.shape[0]

        # Sparse matrix with ones on diagonal
        lambda_diagonal=self.m_lambda*sparse.eye(self.K)
        solve_vecs=np.zeros(shape=(num_solve,self.K))
        for i in range(num_solve):
            if matrix_fixed== 'Q_Matrix':
                R_i=self.Rating_Matrix[i].toarray()
                vec=self.P_Matrix[i]
                N=np.count_nonzero(vec)
                FRu = fixed_vecs.T.dot(R_i.T)
            else:
                R_i=self.Rating_Matrix[:,i].T.toarray()
                vec=self.Q_Matrix[i]
                N=np.count_nonzero(vec)
                FRu=fixed_vecs.T.dot(R_i.T)
            vec_updated=spsolve((vec.dot(vec.T)*sparse.eye(self.K))+lambda_diagonal*N,FRu)
            solve_vecs[i]=vec_updated.T
            if i % 500 ==0:
                print('Fixed:{} Solved 500 vectors in {:0.2f}s'.format(matrix_fixed,time.time()-tic))
        return solve_vecs


    def RMSE(self):
        Predict = self.P_Matrix.dot(self.Q_Matrix.T)
        SEL = (np.asarray((self.Rating_Matrix - Predict) )** 2).sum()
        return np.sqrt(SEL / (self.rows * self.cols))


    def draw(self):
        loops=self.max_loops
        plt.plot(np.arange(0, loops - 1, 1), self.rmse[0:loops - 1], label='RMSE')
        plt.xlabel('loops')
        plt.ylabel('RMSE')
        plt.title('RMSE')
        plt.legend()
        plt.show()

    def loss(self):
        R=self.Rating_Matrix.toarray()
        P=self.P_Matrix
        Q=self.Q_Matrix.T
        Predict = P.dot(Q)
        m = R.shape[0]
        n = R.shape[1]
        k = P.shape[1]
        Loss = 0

        for u in range(0, m - 1):
            for i in range(0, n - 1):
                if R[u, i] == 0:
                    continue
                else:
                    Npu = np.count_nonzero(P, 1)[u]
                    Nqi = np.count_nonzero(Q, 0)[i]
                    P_temp = P[u, :].reshape(1, k)
                    Q_temp = Q[:, i].reshape(k, 1)
                    Loss = Loss + np.square(R[u, i] - Predict[u, i]) + self.m_lambda * (Npu * P_temp.dot(P_temp.transpose()) +Nqi * Q_temp.dot(Q_temp.transpose()))
        return Loss


    def train(self):
        tic=time.time()
        self.rmse=[]
        self.Loss=[]
        self.P_Matrix=np.random.normal(size=(self.rows,self.K))
        self.Q_Matrix=np.random.normal(size=(self.cols,self.K))
        count=0
        rmse=1
        while count<self.max_loops:
            if rmse<self.epsilon:
                print('ALS Completed in loops {}, Time:{:0.2f}s'.format(count, time.time() - tic))
                break
            else:
                print('Update User Matrix for the {} time'.format(count))
                self.P_Matrix=self.iteraton('Q_Matrix',sparse.csr_matrix(self.Q_Matrix))
                print('Update Item Matrix for the {} time'.format(count))
                self.Q_Matrix=self.iteraton('P_Matrix',sparse.csr_matrix(self.P_Matrix))
                rmse=self.RMSE()
                #loss=self.loss()
                print('Loops:{}, RMSE:{:0.3f}'.format(count, rmse))
                self.rmse.append(rmse)
                #self.Loss.append(loss)
                count=count+1
            print('ALS Completed, Time:{:0.2f}s'.format(count, time.time() - tic))
        self.draw()