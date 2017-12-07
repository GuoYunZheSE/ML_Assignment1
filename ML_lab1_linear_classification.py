import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split

# Function sigmoid

def Loss(X,Y,W,b,m_lambda):
    loss=0.5 * m_lambda * W.transpose().dot(W)
    for i in range(X.shape[0]):
        Tensor = Y[i][0] * (W.transpose().dot(X[i]) + b)
        if Tensor < 1:
            loss=loss+1-Tensor
        else:
            loss=loss+0
    return loss


def compare(X,Y,W,b,i):
    if Y[i][0]*(W.transpose().dot(X[i])+b)<1:
        return 1
    else:
        return 0


def Grad(X,Y,W,b,m_lambda):
    grad=m_lambda*W
    for i in range(X.shape[0]):
        grad=grad-(compare(X,Y,W,b,i)*Y[i]*X[i]).reshape(X.shape[1],1)
    return grad

def corrate_rate(X,Y,W):
    count=0
    temp=Y*(X.dot(W)+b)
    for j in temp:
        if j>=1:
            count+=1
        else:
            continue
    rate=count/temp.shape[0]
    return rate


def Draw(loops,train_loss,validation_loss,train_accuracy,val_accuracy):
    #the loss
    plt.plot(np.arange(0,loops-1,1), train_loss[0:loops-1], label='Train Loss')
    plt.plot(np.arange(0,loops-1,1), validation_loss[0:loops-1], label='Validation Loss')
    plt.xlabel('loops')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    #the accuracy
    plt.plot(np.arange(0,loops-1,1), train_accuracy[0:loops-1], label='Train Accuracy')
    plt.plot(np.arange(0,loops-1,1), val_accuracy[0:loops-1], label='Validation Accuracy')
    plt.xlabel('loops')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # Read Data
    Data_Path = 'G:\\2017\\VS2017\\ML_Assignment1\\ML_Assignment1\\DataSet\\australian_scale.txt'
    Data_Parameter, Data_Value = load_svmlight_file(Data_Path)
    Data_Parameter = Data_Parameter.toarray()
    train_X, val_X, train_Y, val_Y = train_test_split(Data_Parameter, Data_Value, test_size=0.25, random_state=1)
    t_row = train_X.shape[0]  # Row Size
    col = train_X.shape[1]  # Column Size
    v_row = val_X.shape[0]
    train_Y = train_Y.reshape(t_row, 1)
    val_Y = val_Y.reshape(v_row, 1)

    # initial W, C, b and our learning rate N
    W = np.random.random(size=(col, 1))
    b=2
    N = 0.000085
    m_lambda = 0.01 #C=1/lambda

    # BGD
    max_loop = 100000  # in case it won't converage
    epsilon = 0.00001
    count = 0
    error = np.zeros((col, 1))
    finish = 0
    TL = []
    VL = []
    train_accuracy=[]
    val_accuracy=[]
    train_right=0
    val_right=0
    while count <= max_loop:
        count += 1
        W = W - N * Grad(train_X, train_Y, W, b, m_lambda)
        if (np.linalg.norm(W - error) < epsilon):
            finish = 1
            break
        else:
            error = W
            Loss_Train = Loss(train_X,train_Y,W,b,m_lambda)
            Loss_Validation = Loss(val_X,val_Y,W,b,m_lambda)
            TL.append(Loss_Train[0] / t_row)
            VL.append(Loss_Validation[0] / v_row)
            train_accuracy.append(corrate_rate(train_X,train_Y,W))
            val_accuracy.append(corrate_rate(val_X,val_Y,W))
            print('Loop {}'.format(count), 'Loss_Train: ', Loss_Train / t_row, 'Loss_Validation: ',
                  Loss_Validation / v_row)
            print('Accuracy: Train: {}, Validation: {}'.format(train_accuracy[count-1],val_accuracy[count-1]))
            # print(W)
    Draw(count,TL,VL,train_accuracy,val_accuracy)
