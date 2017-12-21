import numpy as np
import sklearn
import matplotlib
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
            if i % 100 ==0:
                print('Fixed:{} Solved 1000 vectors in {:0.2f}s'.format(matrix_fixed,time.time()-tic))
        return solve_vecs


    def RMSE(self):
        Predict = self.P_Matrix.dot(self.Q_Matrix.T)
        print(type(self.Rating_Matrix - Predict))
        SEL = ((self.Rating_Matrix - Predict).asarray() ** 2).sum()
        return np.sqrt(SEL / (self.rows * self.cols))


    def train(self):
        tic=time.time()
        self.P_Matrix=np.random.normal(size=(self.rows,self.K))
        self.Q_Matrix=np.random.normal(size=(self.cols,self.K))
        count=0
        rmse=1
        while count<self.max_loops:
            if rmse<self.epsilon:
                print('ALS Completed in loops {}, Time:{:0.2f}'.format(count, time.time() - tic))
                break
            else:
                print('Update User Matrix for the {} time'.format(count))
                self.P_Matrix=self.iteraton('Q_Matrix',sparse.csr_matrix(self.Q_Matrix))
                print('Update Item Matrix for the {} time'.format(count))
                self.Q_Matrix=self.iteraton('P_Matrix',sparse.csr_matrix(self.P_Matrix))
                rmse=self.RMSE()
                print('Loops:{}, RMSE:{:0.5f}'.format(count, rmse))
            print('ALS Completed, Time:{:0.2f}'.format(count, time.time() - tic))