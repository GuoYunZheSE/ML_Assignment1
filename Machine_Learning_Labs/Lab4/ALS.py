import numpy as np
import sklearn
import matplotlib.pyplot as plt
import time
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


class ALS:


    def __init__(self,R,W,TR,TW,Parameters):
        self.Rating_Matrix=R
        self.rows=Parameters['Row']
        self.cols=Parameters['Cols']
        self.K=Parameters['K']
        self.m_lambda=Parameters['Lambda']
        self.max_loops=Parameters['Max Loops']
        self.epsilon=Parameters['epsilon']
        self.Weight_Matrix=W
        self.Test_Rating_Matrix=TR
        self.Test_Weight_Matrix=TW


    def iteraton(self, matrix_fixed):
        tic=time.time()
        if matrix_fixed== 'P_Matrix':
            solve_vecs=np.linalg.solve(np.dot(self.P_Matrix.T,self.P_Matrix)+self.m_lambda*np.eye(self.K),
                                       np.dot(self.P_Matrix.T,self.Rating_Matrix)).T
        else:
            solve_vecs=np.linalg.solve(np.dot(self.Q_Matrix.T,self.Q_Matrix)+self.m_lambda*np.eye(self.K),
                                        np.dot(self.Q_Matrix.T,self.Rating_Matrix.T)).T
        print('Update {} Successfully. Time:{:0.5f}s'.format(matrix_fixed,time.time()-tic))
        return solve_vecs


    def RMSE(self,Rating_Matrix):
        Predict = self.P_Matrix.dot(self.Q_Matrix.T)
        SEL = (np.asarray((Rating_Matrix - Predict) )** 2).sum()
        return np.sqrt(SEL / (self.rows * self.cols))


    def draw(self):
        loops=self.max_loops
        plt.plot(np.arange(0, loops - 1, 1), self.train_rmse[0:loops - 1], label='Train_RMSE')
        plt.plot(np.arange(0, loops - 1, 1), self.test_rmse[0:loops - 1], label='Test_RMSE')
        plt.xlabel('loops')
        plt.ylabel('RMSE')
        plt.title('RMSE')
        plt.legend()
        plt.show()

        loops=self.max_loops
        plt.plot(np.arange(0, loops - 1, 1), self.test_Loss[0:loops - 1], label='Test_Loss')
        plt.plot(np.arange(0, loops - 1, 1), self.train_Loss[0:loops - 1], label='Train_Loss')
        plt.xlabel('loops')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.show()


    def loss(self,Rating_Matrix,Weight_Matrix):
        Predict=self.P_Matrix.dot(self.Q_Matrix.T)
        SEL = np.sum((Weight_Matrix * np.asarray(Rating_Matrix - Predict))**2)
        RP=0
        RQ=0
        for i in range(0,self.rows):
            RP+=self.m_lambda*self.P_Matrix[i,:].dot(self.P_Matrix[i,:].T)
        for i in range(0,self.cols):
            RQ+=self.m_lambda*self.Q_Matrix[i,:].dot(self.Q_Matrix[i,:].T)
        LOSS=SEL+RP+RQ
        return LOSS/(self.rows*self.cols)



    def train(self):
        tic=time.time()
        self.train_rmse=[]
        self.train_Loss=[]
        self.test_rmse=[]
        self.test_Loss=[]
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
                self.P_Matrix=self.iteraton('Q_Matrix')
                print('Update Item Matrix for the {} time'.format(count))
                self.Q_Matrix=self.iteraton('P_Matrix')

                train_rmse=self.RMSE(self.Rating_Matrix)
                train_loss=self.loss(self.Rating_Matrix,self.Weight_Matrix)
                test_rmse=self.RMSE(self.Test_Rating_Matrix)
                test_loss=self.loss(self.Test_Rating_Matrix,self.Test_Weight_Matrix)

                self.train_rmse.append(train_rmse)
                self.train_Loss.append(train_loss)
                self.test_rmse.append(test_rmse)
                self.test_Loss.append(test_loss)

                print('Loops:{} Completed. Train: RMSE:{:0.2f},Loss:{:0.2f}; Test: RMSE:{:0.2f},Loss:{:0.2f}'
                      .format(count, train_rmse,train_loss,test_rmse,test_loss))

                count=count+1
            print('ALS Completed, Time:{:0.2f}s'.format(time.time() - tic))