import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split


def Grad(m_lambda,W,train_X):
    grad = m_lambda * W + (np.dot(train_X.transpose(), np.dot(train_X, W)))
    return grad


if __name__ == '__main__':
    # Cost Function:0.5*lambda*W'*W+0.5*(Y-X*W)'*(Y-X*W)

    # Read Data
    Data_Path='G:\\2017\\VS2017\\ML_Assignment1\\ML_Assignment1\\DataSet\\housing.txt'
    Data_Parameter,Data_Value=load_svmlight_file(Data_Path)
    Data_Parameter=Data_Parameter.toarray()
    train_X, val_X,train_Y,val_Y = train_test_split(Data_Parameter,Data_Value,test_size=0.3, random_state=1)
    t_row=train_X.shape[0]#Row Size
    col=train_X.shape[1]#Column Size
    v_row=val_X.shape[0]
    train_Y=train_Y.reshape(t_row, 1)
    val_Y=val_Y.reshape(v_row, 1)

    # initial W, lambda and our learning rate N
    W = np.random.random(size=(col, 1))
    m_lambda = 0.01
    N = 0.000085




    # BGD
    max_loop = 100000  # in case it won't converage
    epsilon = 0.000001
    count = 0
    error = np.zeros((col, 1))
    finish = 0
    Tensor = np.dot(-train_X.transpose(), train_Y)  # It never change during our process, so I put it in a tensor
    while count <= max_loop:
        count += 1
        W = W - N * (Grad(m_lambda,W,train_X) + Tensor)
        if (np.linalg.norm(W - error) < epsilon):
            finish = 1
            break
        else:
            error = W
            Loss_Train=0.5*m_lambda*W.transpose().dot(W)+0.5*(train_Y-train_X.dot(W)).transpose().dot((train_Y-train_X.dot(W)))
            Loss_Validation=0.5*m_lambda*W.transpose().dot(W)+0.5*(val_Y-val_X.dot(W)).transpose().dot((val_Y-val_X.dot(W)))
            print('Loop {}'.format(count),'Loss_Train: ',Loss_Train,'Loss_Validation: ', Loss_Validation)
            # print(count)  #You can choose whether to print Count/W
            # print(W)