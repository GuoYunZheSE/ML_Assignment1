import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split


def Draw(title,loops,train_loss,validation_loss,train_accuracy,val_accuracy,test_accuracy):
    #the loss
    plt.plot(np.arange(0,loops-1,1), train_loss[0:loops-1], label='Train Loss')
    plt.plot(np.arange(0,loops-1,1), validation_loss[0:loops-1], label='Validation Loss')
    plt.xlabel('loops')
    plt.ylabel('loss')
    plt.title('{} Loss'.format(title))
    plt.legend()
    plt.show()

    #the accuracy
    plt.plot(np.arange(0,loops-1,1), train_accuracy[0:loops-1], label='Train Accuracy')
    plt.plot(np.arange(0,loops-1,1), val_accuracy[0:loops-1], label='Validation Accuracy')
    plt.plot(np.arange(0, loops - 1, 1), test_accuracy[0:loops - 1], label='Test Accuracy')
    plt.xlabel('loops')
    plt.ylabel('Accuracy')
    plt.title('{} Accuracy'.format(title))
    plt.legend()
    plt.show()


def DataChange(filepath):
    # To change the -1 values in Dataset to 0
    date=''
    with open(filepath,'r+') as fr:
        for eachline in fr.readlines():
            if '-1' in eachline:
                eachline=eachline.replace('-1','+0')
            date=date+eachline
    with open(filepath,'r+') as fw:
        fw.writelines(date)


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
    temp=Y.transpose().dot(np.log(hypothesis(W,X)))+(1-Y.transpose()).dot(np.log(1-hypothesis(W,X)))
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
            p[it, 0] = 0
        if p[it,0]==Y[it][0]:
            hit=hit+1
    return hit/m


def SGD(Parameters):
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
    count = 0
    error = np.zeros((col, 1))
    TL=[]#train loss
    VL=[]#validation loss
    train_accuracy=[]
    val_accuracy=[]
    test_accutacy=[]
    while count <= max_loop:
        i = random.randint(0, train_X.shape[0] - 1)
        grad = (hypothesis(W, train_X[i]) - train_Y[i]) * train_X[i]
        grad = grad.reshape(train_X.shape[1], 1)
        count = count + 1
        W = W - N * grad
        if np.linalg.norm(W - error) < epsilon:
            break
        else:
            error = W
            Loss_Train = loss(train_X, train_Y, W)
            Loss_Validation = loss(val_X, val_Y, W)
            TL.append(Loss_Train[0])
            VL.append(Loss_Validation[0])
            train_accuracy.append(accuracy(train_X, train_Y, W, threshold))
            val_accuracy.append(accuracy(val_X, val_Y, W, threshold))
            test_accutacy.append(accuracy(Test_X, Test_Y, W, threshold))
            print('Loop {}'.format(count), 'Loss_Train: ', Loss_Train, 'Loss_Validation: ',
                    Loss_Validation)
            print('Accuracy: Train: {}, Validation: {}, Test: {}'.format(train_accuracy[count - 1],
                                                                             val_accuracy[count - 1],
                                                                             test_accutacy[count - 1]))
    print('SGD Completed. Time Used:{}'.format(time.time()-tic))
    Draw('SGD',count,TL,VL,train_accuracy,val_accuracy,test_accutacy)
    return VL,test_accutacy

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
    gamma=Parameters['decoy_rate']
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
    Draw('Momentum',count,TL,VL,train_accuracy,val_accuracy,test_accutacy)
    print('Momentum Completed Successfully. Time used:{:.2f}'.format(time.time()-tic))
    return VL,test_accutacy

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
    gamma=Parameters['decoy_rate']
    count = 0
    error = np.zeros((train_X.shape[1], 1))
    velocity=np.zeros((train_X.shape[1],1))
    velocity_prev = np.zeros((train_X.shape[1], 1))
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
        velocity_prev=velocity
        velocity=gamma*velocity-N*grad
        # Gradient decent
        W=W-gamma*velocity_prev+(1+gamma)*velocity

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
    Draw('NAG',count,TL,VL,train_accuracy,val_accuracy,test_accutacy)
    print('Momentum Completed Successfully. Time used:{:.2f}'.format(time.time()-tic))
    return VL, test_accutacy

def Adagrad(Parameters):
    # This method is NGA with the adaptive learning rate method: Adagrad
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

    # Initialize
    count = 0
    error = np.zeros((train_X.shape[1], 1))
    cache=np.zeros((train_X.shape[1],1))
    eps=0.000001
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
        # Compute Cache
        cache=cache+(grad**2)
        # Gradient decent
        W=W-N*grad/(np.sqrt(cache)+eps)

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
    Draw('Adagrad',count,TL,VL,train_accuracy,val_accuracy,test_accutacy)
    print('Adagrad Completed Successfully. Time used:{:.2f}'.format(time.time()-tic))
    return VL, test_accutacy

def AdaDelta(Parameters):
    # Load parameters from Parameters
    tic = time.time()
    train_X = Parameters['Train_X']
    train_Y = Parameters['Train_Y']
    val_X = Parameters['Val_X']
    val_Y = Parameters['Val_Y']
    W = Parameters['Weights']
    max_loop = Parameters['Max_Loops']
    epsilon = Parameters['Epsilon']
    threshold = Parameters['threshold']
    Test_X = Parameters['Test_X']
    Test_Y = Parameters['Test_Y']
    decay_rate=Parameters['decoy_rate']

    # Initialize
    count = 0
    error = np.zeros((train_X.shape[1], 1))
    cache = np.zeros((train_X.shape[1], 1))
    delt  = np.zeros((train_X.shape[1], 1))
    eps = 0.000001
    TL = []  # train loss
    VL = []  # validation loss
    train_accuracy = []
    val_accuracy = []
    test_accutacy = []

    while count <= max_loop:
        count = count + 1
        i = random.randint(0, train_X.shape[0] - 1)

        # Compute grad
        grad = (hypothesis(W, train_X[i]) - train_Y[i]) * train_X[i]
        grad = grad.reshape(train_X.shape[1], 1)

        # Compute Cache
        cache = decay_rate*cache +(1-decay_rate)*(grad ** 2)

        # Gradient decent
        deltW = -np.sqrt(delt+eps)*grad / (np.sqrt(cache) + eps)
        W=W+deltW
        delt  =decay_rate*delt+(1-decay_rate)*deltW**2
        # Use relative error to decide whether to stop
        if np.linalg.norm(W - error) / Max(W, error) < epsilon:
            break
        else:
            error = W
            Loss_Train = loss(train_X, train_Y, W)
            Loss_Validation = loss(val_X, val_Y, W)
            TL.append(Loss_Train[0])
            VL.append(Loss_Validation[0])
            train_accuracy.append(accuracy(train_X, train_Y, W, threshold))
            val_accuracy.append(accuracy(val_X, val_Y, W, threshold))
            test_accutacy.append(accuracy(Test_X, Test_Y, W, threshold))
            print('Loop {}'.format(count), 'Loss_Train: ', Loss_Train, 'Loss_Validation: ',
                  Loss_Validation)
            print('Accuracy: Train: {}, Validation: {}, Test: {}'.format(train_accuracy[count - 1],
                                                                         val_accuracy[count - 1],
                                                                         test_accutacy[count - 1]))
    Draw('AdaDelta',count, TL, VL, train_accuracy, val_accuracy, test_accutacy)
    print('AdaDelta Completed Successfully. Time used:{:.2f}'.format(time.time() - tic))
    return VL, test_accutacy

def RMSprop(Parameters):
    # This method is NGA with the adaptive learning rate method: Adagrad
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
    decay_rate=Parameters['decoy_rate']

    # Initialize
    count = 0
    error = np.zeros((train_X.shape[1], 1))
    cache=np.zeros((train_X.shape[1],1))
    eps=0.000001
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
        # Compute Cache
        cache=decay_rate*cache+(1-decay_rate)*(grad**2)
        # Gradient decent
        W=W-N*grad/(np.sqrt(cache)+eps)

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
    Draw('RMSprop',count,TL,VL,train_accuracy,val_accuracy,test_accutacy)
    print('RMSprop Completed Successfully. Time used:{:.2f}'.format(time.time()-tic))
    return VL, test_accutacy

def Adam(Parameters):
    # This method is NGA with the adaptive learning rate method: Adagrad
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
    eps=Parameters['eps']
    beta1=Parameters['Beta1']
    beta2 = Parameters['Beta2']

    # Initialize
    count = 0
    error = np.zeros((train_X.shape[1], 1))
    m = np.zeros((train_X.shape[1],1))
    v = np.zeros((train_X.shape[1], 1))
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

        # Compute
        m=beta1*m+(1-beta1)*grad
        mt=m/(1-beta1**count)
        v=beta2*v+(1-beta2)*(grad**2)
        vt=v/(1-beta2**count)
        # Gradient decent
        W=W-N*mt/(np.sqrt(vt)+eps)

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
    Draw('Adam',count,TL,VL,train_accuracy,val_accuracy,test_accutacy)
    print('Adam Completed Successfully. Time used:{:.2f}'.format(time.time()-tic))
    return VL, test_accutacy


if __name__ == '__main__':
    # Read Data
    tic=time.time()
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
               'Learning_Rate':0.01,
               'Max_Loops':5000,
               'Epsilon':0.00000001,
               'threshold':0.4,
               'decoy_rate':0.9,
               'eps':0.00000001,
               'Beta1':0.9,
               'Beta2':0.999
               }
    SGD_VL,SGD_test_accuracy=SGD(Parameter)
    Momentum_VL,Momentum_test_accuracy=Momentum(Parameter)
    NAG_VL,NAG_test_accuracy=NAG(Parameter)
    Adagrad_VL,Adagrad_test_accuracy=Adagrad(Parameter)
    AdaDelta_VL,AdaDelta_test_accuracy=AdaDelta(Parameter)
    Adam_VL,Adam_test_accuracy=Adam(Parameter)
    print('All Time Used:{:0.2f}s'.format(time.time()-tic))

    plt.plot(np.arange(0,Parameter['Max_Loops']-1,1), SGD_VL[0:Parameter['Max_Loops']-1], label='SGD')
    plt.plot(np.arange(0,Parameter['Max_Loops']-1,1), Momentum_VL[0:Parameter['Max_Loops']-1], label='Momentum')
    plt.plot(np.arange(0, Parameter['Max_Loops'] - 1, 1), NAG_VL[0:Parameter['Max_Loops'] - 1], label='NAG')
    plt.plot(np.arange(0, Parameter['Max_Loops'] - 1, 1), Adagrad_VL[0:Parameter['Max_Loops'] - 1], label='Adagrad')
    plt.plot(np.arange(0, Parameter['Max_Loops'] - 1, 1), AdaDelta_VL[0:Parameter['Max_Loops'] - 1], label='AdaDelta')
    plt.plot(np.arange(0, Parameter['Max_Loops'] - 1, 1), Adam_VL[0:Parameter['Max_Loops'] - 1], label='Adam')
    plt.xlabel('loops')
    plt.ylabel('loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.show()


    plt.plot(np.arange(0,Parameter['Max_Loops']-1,1), SGD_test_accuracy[0:Parameter['Max_Loops']-1], label='SGD')
    plt.plot(np.arange(0,Parameter['Max_Loops']-1,1), Momentum_test_accuracy[0:Parameter['Max_Loops']-1], label='Momentum')
    plt.plot(np.arange(0, Parameter['Max_Loops'] - 1, 1), NAG_test_accuracy[0:Parameter['Max_Loops'] - 1], label='NAG')
    plt.plot(np.arange(0, Parameter['Max_Loops'] - 1, 1), Adagrad_test_accuracy[0:Parameter['Max_Loops'] - 1], label='Adagrad')
    plt.plot(np.arange(0, Parameter['Max_Loops'] - 1, 1), AdaDelta_test_accuracy[0:Parameter['Max_Loops'] - 1], label='AdaDelta')
    plt.plot(np.arange(0, Parameter['Max_Loops'] - 1, 1), Adam_test_accuracy[0:Parameter['Max_Loops'] - 1], label='Adam')
    plt.xlabel('loops')
    plt.ylabel('loss')
    plt.title('Test Accuracy')
    plt.legend()
    plt.show()