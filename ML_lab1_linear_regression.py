import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split


def Grad(m_lambda,W,train_X):
    grad = m_lambda * W + (np.dot(train_X.transpose(), np.dot(train_X, W)))
    return grad


def Draw(loops,train_loss,validation_loss):
    # the first 100loops
    print('Drawing...')
    plt.plot(np.arange(0,200,1), train_loss[0:200], label='Train Loss')
    plt.plot(np.arange(0,200,1), validation_loss[0:200], label='Validation Loss')
    plt.xlabel('loops')
    plt.ylabel('loss')
    plt.title('The First 200 loops')
    plt.legend()
    plt.show()

    # the last 10000 loops
    plt.plot(np.arange(201, loops-1, 1), train_loss[201:loops-1], label='Train Loss')
    plt.plot(np.arange(201, loops-1, 1), validation_loss[201:loops-1], label='Validation Loss')
    plt.xlabel('loops')
    plt.ylabel('loss')
    plt.title('The rest loops')
    plt.legend()
    plt.show()
    print('Draw Completed')


def BGD(parameters):
    tic1=time.time()
    epsilon = parameters['epsilon']
    max_loop=parameters['max_loops']
    m_lambda=parameters['lambda']
    N=parameters['learning_rate']
    train_X=parameters['train_X']
    train_Y=parameters['train_Y']
    val_X=parameters['val_X']
    val_Y=parameters['val_Y']
    W=parameters['Weights']
    count = 0
    error = np.zeros((col, 1))
    finish = 0
    Tensor = np.dot(-train_X.transpose(), train_Y)  # It never change during our process, so I put it in a tensor to
                                                    # decrease computation cost
    TL=[]
    VL=[]
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
            TL.append(Loss_Train[0]/train_X.shape[0])
            VL.append(Loss_Validation[0]/val_X.shape[0])
            print('Loop {}'.format(count),'Loss_Train: ',Loss_Train/train_X.shape[0],'Loss_Validation: ', Loss_Validation/val_X.shape[0])
            # print(count)  #You can choose whether to print Count/W
            # print(W)
    print('BGD Completed Successfully. Time Used:{}s'.format(time.time()-tic1))
    return count,TL,VL


if __name__ == '__main__':
    # Cost Function:0.5*lambda*W'*W+0.5*(Y-X*W)'*(Y-X*W)

    # Read Data
    tic0=time.time()
    Data_Path='/home/lucas/Codes/GitHub/ML_Assignment1/ML_Assignment1/DataSet/housing.txt'
    Data_Parameter,Data_Value=load_svmlight_file(Data_Path)
    Data_Parameter=Data_Parameter.toarray()
    train_X, val_X,train_Y,val_Y = train_test_split(Data_Parameter,Data_Value,test_size=0.3, random_state=1)
    t_row=train_X.shape[0]#Row Size
    col=train_X.shape[1]#Column Size
    v_row=val_X.shape[0]
    train_Y=train_Y.reshape(t_row, 1)
    val_Y=val_Y.reshape(v_row, 1)

    W = np.random.random(size=(col, 1))
    Paramaters={
        'Weights': W,
        'train_X':train_X,
        'train_Y':train_Y,
        'val_X':val_X,
        'val_Y':val_Y,
        'max_loops':100000,  # in case it won't converage
        'epsilon':0.000001,  # if Wt+1-Wt<epsilon then stop BGD
        'lambda':0.1,  # regularize
        'learning_rate':0.000085,  #learning rate
    }
    loops,train_loss,val_loss=BGD(Paramaters)
    Draw(loops,train_loss,val_loss)
    print('Maching Learning Lab-1:linear regression Completed Successfully. Time Used:{}s'.format(time.time()-tic0))