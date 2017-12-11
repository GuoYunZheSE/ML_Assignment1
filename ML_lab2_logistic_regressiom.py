import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split


def Draw(loops,train_loss,validation_loss,train_accuracy,val_accuracy,test_accuracy):
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
    plt.plot(np.arange(0, loops - 1, 1), test_accuracy[0:loops - 1], label='Test Accuracy')
    plt.xlabel('loops')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()


def Max(M1,M2): # Return the Matrix with a bigger norm
    N1=np.linalg.norm(M1)
    N2=np.linalg.norm(M2)
    if N1>=N2:
        return N1
    else:
        return N2


def sigmoid(X):
    sig=1/(1+np.exp(-X))
    return sig

def hypothesis(W,X):
    return sigmoid(X.dot(W))


def loss(X,Y,W):
    n=X.shape[0]
    temp=Y.transpose().dot(np.log10(hypothesis(W,X)))+(1-Y.transpose()).dot(np.log10(1-hypothesis(W,X)))
    loss=(-1*temp)/n
    return loss


def accuracy(X,Y,W,threshold):
    m,n=X.shape
    p = np.zeros(shape=(m, 1))
    h=hypothesis(W,X)
    hit=0
    for it in range(0, h.shape[0]):
        if h[it] >= threshold:
            p[it, 0] = 1
        else:
            p[it, 0] = -1
        if p[it,0]==Y[it][0]:
            hit=hit+1
    return hit/m


def SGD(Parameters):
    train_X=Parameters['Train_X']
    train_Y=Parameters['Train_Y']
    val_X=Parameters['Val_X']
    val_Y=Parameters['Val_Y']
    W=Parameters['Weights']
    N=Parameters['Learning_Rate']
    max_loop = Parameters['Max_Loops']
    epsilon = Parameters['Epsilon']
    threshold=Parameters['threshold']
    Test_X=Parameters['Test_X']
    Test_Y=Parameters['Test_Y']
    count = 0
    error = np.zeros((col, 1))
    TL=[]#train loss
    VL=[]#validation loss
    train_accuracy=[]
    val_accuracy=[]
    test_accutacy=[]
    while count <= max_loop:
        i=random.randint(0,train_X.shape[0]-1)
        grad=(hypothesis(W,train_X[i])-train_Y[i])*train_X[i]
        grad=grad.reshape(train_X.shape[1],1)
        count =count+1
        W = W - N * grad
        if np.linalg.norm(W - error) < epsilon:
            break
        else:
            error = W
            Loss_Train = loss(train_X,train_Y,W)
            Loss_Validation = loss(val_X,val_Y,W)
            TL.append(Loss_Train[0])
            VL.append(Loss_Validation[0])
            train_accuracy.append(accuracy(train_X,train_Y,W,threshold))
            val_accuracy.append(accuracy(val_X,val_Y,W,threshold))
            test_accutacy.append(accuracy(Test_X,Test_Y,W,threshold))
            print('Loop {}'.format(count), 'Loss_Train: ', Loss_Train, 'Loss_Validation: ',
                  Loss_Validation)
            print('Accuracy: Train: {}, Validation: {}, Test: {}'.format(train_accuracy[count-1],val_accuracy[count-1],test_accutacy[count-1]))
    Draw(count,TL,VL,train_accuracy,val_accuracy,test_accutacy)


def Momentum(Parameters):
    tic=time.time()
    train_X=Parameters['Train_X']
    train_Y=Parameters['Train_Y']
    val_X=Parameters['Val_X']
    val_Y=Parameters['Val_Y']
    W=Parameters['Weights']
    N=Parameters['Learning_Rate']
    max_loop = Parameters['Max_Loops']
    epsilon = Parameters['Epsilon']
    threshold=Parameters['threshold']
    Test_X=Parameters['Test_X']
    Test_Y=Parameters['Test_Y']
    gamma=Parameters['gamma']
    count = 0
    error = np.zeros((train_X.shape[1], 1))
    velocity=np.zeros((train_X.shape[1],1))
    TL=[]#train loss
    VL=[]#validation loss
    train_accuracy=[]
    val_accuracy=[]
    test_accutacy=[]
    while count <= max_loop:
        count = count + 1
        i=random.randint(0,train_X.shape[0]-1)

        # Compute grad
        grad=(hypothesis(W,train_X[i])-train_Y[i])*train_X[i]
        grad=grad.reshape(train_X.shape[1],1)
        # Compute velocity
        velocity=gamma*velocity+N*grad
        # Gradient decent
        W=W-velocity

        if np.linalg.norm(W - error) < epsilon:
            break
        else:
            error = W
            Loss_Train = loss(train_X,train_Y,W)
            Loss_Validation = loss(val_X,val_Y,W)
            TL.append(Loss_Train[0])
            VL.append(Loss_Validation[0])
            train_accuracy.append(accuracy(train_X,train_Y,W,threshold))
            val_accuracy.append(accuracy(val_X,val_Y,W,threshold))
            test_accutacy.append(accuracy(Test_X,Test_Y,W,threshold))
            print('Loop {}'.format(count), 'Loss_Train: ', Loss_Train, 'Loss_Validation: ',
                  Loss_Validation)
            print('Accuracy: Train: {}, Validation: {}, Test: {}'.format(train_accuracy[count-1],val_accuracy[count-1],test_accutacy[count-1]))
    Draw(count,TL,VL,train_accuracy,val_accuracy,test_accutacy)
    print('Momentum Completed Successfully. Time used:{:.2f}'.format(time.time()-tic))


def NAG(Parameters):
    tic=time.time()
    train_X=Parameters['Train_X']
    train_Y=Parameters['Train_Y']
    val_X=Parameters['Val_X']
    val_Y=Parameters['Val_Y']
    W=Parameters['Weights']
    N=Parameters['Learning_Rate']
    max_loop = Parameters['Max_Loops']
    epsilon = Parameters['Epsilon']
    threshold=Parameters['threshold']
    Test_X=Parameters['Test_X']
    Test_Y=Parameters['Test_Y']
    gamma=Parameters['gamma']
    count = 0
    error = np.zeros((train_X.shape[1], 1))
    velocity=np.zeros((train_X.shape[1],1))
    TL=[]#train loss
    VL=[]#validation loss
    train_accuracy=[]
    val_accuracy=[]
    test_accutacy=[]
    while count <= max_loop:
        count = count + 1
        i=random.randint(0,train_X.shape[0]-1)

        # Compute grad
        grad=(hypothesis(W,train_X[i])-train_Y[i])*train_X[i]
        grad=grad.reshape(train_X.shape[1],1)
        # Compute velocity
        velocity=gamma*velocity+N*grad
        # Gradient decent
        W=W-velocity

        # Use relative error to decide whether to stop
        if np.linalg.norm(W - error)/Max(W,error) < epsilon:
            break
        else:
            error = W
            Loss_Train = loss(train_X,train_Y,W)
            Loss_Validation = loss(val_X,val_Y,W)
            TL.append(Loss_Train[0])
            VL.append(Loss_Validation[0])
            train_accuracy.append(accuracy(train_X,train_Y,W,threshold))
            val_accuracy.append(accuracy(val_X,val_Y,W,threshold))
            test_accutacy.append(accuracy(Test_X,Test_Y,W,threshold))
            print('Loop {}'.format(count), 'Loss_Train: ', Loss_Train, 'Loss_Validation: ',
                  Loss_Validation)
            print('Accuracy: Train: {}, Validation: {}, Test: {}'.format(train_accuracy[count-1],val_accuracy[count-1],test_accutacy[count-1]))
    Draw(count,TL,VL,train_accuracy,val_accuracy,test_accutacy)
    print('Momentum Completed Successfully. Time used:{:.2f}'.format(time.time()-tic))

if __name__ == '__main__':
    # Read Data
    Data_Path = '/home/lucas/Codes/GitHub/ML_Assignment1/ML_Assignment1/DataSet/a9a.txt'
    Test_Path='/home/lucas/Codes/GitHub/ML_Assignment1/ML_Assignment1/DataSet/a9a.t'
    Data_Parameter, Data_Value = load_svmlight_file(Data_Path)
    Test_Parameter,Test_Value=load_svmlight_file(Test_Path)
    Test_Parameter=Test_Parameter.toarray()
    Test_Parameter=np.hstack([Test_Parameter,np.zeros(shape=(Test_Parameter.shape[0],1))])
    Test_Value=Test_Value.reshape(Test_Value.shape[0],1)
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
               'Test_X':Test_Parameter,
               'Test_Y':Test_Value,
               'Weights':W,
               'Learning_Rate':0.0025,
               'Max_Loops':2500,
               'Epsilon':0.000001,
               'threshold':0.4,
               'gamma':0.9
               }
    Momentum(Parameter)