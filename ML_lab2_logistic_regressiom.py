import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split


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



def sigmoid(X):
    sig=np.exp(X)/(1+np.exp(X))
    return sig


def loss(X,Y,W):
    n=X.shape[0]
    temp=Y*np.log(sigmoid(X.dot(W)))+(1-Y)*np.log(1-sigmoid(X.dot(W)))
    loss=-temp/n
    return loss


def accuracy(X,Y,W,threshold):
    m,n=X.shape
    p = np.zeros(shape=(m, 1))
    h=sigmoid(X.dot(W))
    count=0
    for it in range(0, h.shape[0]):
        if h[it] >= threshold:
            p[it, 0] = 1
        else:
            p[it, 0] = 0
        if p[it,0]==Y[it][0]:
            count=+1
    return count/m


def SGD(Parameters):
    train_X=Parameters['Train_X']
    train_Y=Parameters['Train_Y']
    val_X=Parameters['Val_X']
    val_Y=Parameters['Val_Y']
    W=Parameters['Weights']
    N=Parameters['Learning_Rate']
    max_loop = Parameters['Max_Loops']
    epsilon = Parameters['Epsilon']
    count = 0
    error = np.zeros((col, 1))
    TL=[]#train loss
    VL=[]#validation loss
    train_accuracy=[]
    val_accuracy=[]
    while count <= max_loop:
        i=random.randint(0,train_X.shape[0])
        grad = (sigmoid(train_X[i].dot(W)) - train_Y[i]).dot(train_X[i])
        count += 1
        W = W - N * grad

        if (np.linalg.norm(W - error) < epsilon):
            break
        else:
            error = W
            Loss_Train = loss(train_X,train_Y,W)
            Loss_Validation = loss(val_X,val_Y,W)
            TL.append(Loss_Train[0] / train_X.shape[0])
            VL.append(Loss_Validation[0] / val_X.shape[0])
            train_accuracy.append(accuracy(train_X,train_Y,W))
            val_accuracy.append(accuracy(val_X,val_Y,W))
            print('Loop {}'.format(count), 'Loss_Train: ', Loss_Train / t_row, 'Loss_Validation: ',
                  Loss_Validation / v_row)
            print('Accuracy: Train: {}, Validation: {}'.format(train_accuracy[count-1],val_accuracy[count-1]))
    Draw(count,TL,VL,train_accuracy,val_accuracy)

if __name__ == '__main__':
    # Read Data
    Data_Path = '/home/lucas/Codes/GitHub/ML_Assignment1/ML_Assignment1/DataSet/a9a.txt'
    Data_Parameter, Data_Value = load_svmlight_file(Data_Path)
    Data_Parameter = Data_Parameter.toarray()
    train_X, val_X, train_Y, val_Y = train_test_split(Data_Parameter, Data_Value, test_size=0.3, random_state=1)
    t_row = train_X.shape[0]  # Row Size
    col = train_X.shape[1]  # Column Size
    v_row = val_X.shape[0]
    train_Y = train_Y.reshape(t_row, 1)
    val_Y = val_Y.reshape(v_row, 1)
    W = np.random.random(size=(col, 1))
    Parameter={'Train_X':train_X,
               'Train_Y':train_Y,
               'Val_X':val_X,
               'Val_Y':val_Y,
               'Weights':W,
               'Learning_Rate':0.000001,
               'Max_Loops':50000,
               'Epsilon':0.00001}
    SGD(Parameter)