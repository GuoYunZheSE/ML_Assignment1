import numpy as np
from sklearn.cross_validation import train_test_split
import time
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from ALS import ALS

def loadMovieLens(parametes,directory_path):
    tic=time.time()
    rows=parametes['Row']
    cols=parametes['Cols']
    mode=parametes['Mode']
    # Method:Hold-Out
    if mode==0:
        file_path=directory_path+'/u.data'
        Train_Rating_Matrix=np.zeros(shape=(rows,cols))
        Train_Weight_Matrix = np.zeros(shape=(rows, cols))
        Test_Rating_Matrix=np.zeros(shape=(rows,cols))
        Test_Weight_Matrix = np.zeros(shape=(rows, cols))
        origin_data=np.loadtxt(file_path)
        train,test=train_test_split(origin_data,test_size=0.2,random_state=1)
        for i in range(0,train.shape[0]):
            user_id=int(train[i][0])
            item_id=int(train[i][1])
            rating=train[i][2]
            Train_Rating_Matrix[user_id-1,item_id-1]=rating
            if rating!=0:
                Train_Weight_Matrix[user_id - 1, item_id - 1] = rating
        for i in range(0,test.shape[0]):
            user_id=int(test[i][0])
            item_id=int(test[i][1])
            rating=test[i][2]
            Test_Rating_Matrix[user_id-1,item_id-1]=rating
            if rating!=0:
                Test_Weight_Matrix[user_id - 1, item_id - 1] = rating
        print('Loading MovieLen Successfully. Time Used:{:0.2f}. Mode:0'.format(time.time()-tic))
        return Train_Rating_Matrix,Train_Weight_Matrix,Test_Rating_Matrix,Test_Weight_Matrix

    # Method Cross validation
    if mode==1:
        Train_Rating_Matrixs=[]
        Train_Weight_Matrixs=[]
        Test_Rating_Matrixs=[]
        Test_Weight_Matrixs=[]
        for i in range(1,6):
            Train_Rating_Matrix = np.zeros(shape=(rows, cols))
            Train_Weight_Matrix = np.zeros(shape=(rows, cols))
            Test_Rating_Matrix = np.zeros(shape=(rows, cols))
            Test_Weight_Matrix = np.zeros(shape=(rows, cols))
            train_path = directory_path + '/u{}.base'.format(i)
            test_path = directory_path + '/u{}.test'.format(i)
            train_data = np.loadtxt(train_path)
            test_data = np.loadtxt(test_path)
            for i in range(0, train_data.shape[0]):
                user_id = int(train_data[i][0])
                item_id = int(train_data[i][1])
                rating = train_data[i][2]
                Train_Rating_Matrix[user_id - 1, item_id - 1] = rating
                if rating != 0:
                    Train_Weight_Matrix[user_id - 1, item_id - 1] = rating
            for i in range(0, test_data.shape[0]):
                user_id = int(test_data[i][0])
                item_id = int(test_data[i][1])
                rating = test_data[i][2]
                Test_Rating_Matrix[user_id - 1, item_id - 1] = rating
                if rating != 0:
                    Test_Weight_Matrix[user_id - 1, item_id - 1] = rating
            Train_Rating_Matrixs.append(Train_Rating_Matrix)
            Train_Weight_Matrixs.append(Train_Weight_Matrix)
            Test_Rating_Matrixs.append(Test_Rating_Matrix)
            Test_Weight_Matrixs.append(Test_Weight_Matrix)
        print('Loading MovieLen Successfully. Time Used:{:0.2f}. Mode:1'.format(time.time()-tic))
        Data=[Train_Rating_Matrixs,Train_Weight_Matrixs,Test_Rating_Matrixs,Test_Rating_Matrixs]
        return Data

def draw(loops,dic,x_label,y_label,title):
    loops = loops
    for m_label,list in dic.items():
        plt.plot(np.arange(0, loops - 1, 1), list[0:loops - 1], label=m_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dic_path='/home/lucas/Codes/GitHub/ML_Assignment1/ML_Assignment1/DataSet/ml-100k'
    Parameters={
        'Row':943,
        'Cols':1682,
        'Mode':1,
        'K':40,
        'Lambda':0.1,
        'Max Loops':100,
        'epsilon':0.00001
    }
    if Parameters['Mode']==0:
        Train_Rating_Matrix, Train_Weight_Matrix, Test_Rating_Matrix, Test_Weight_Matrix = loadMovieLens(Parameters,
                                                                                                         dic_path)
        ALS_0 = ALS(Train_Rating_Matrix, Train_Weight_Matrix, Test_Rating_Matrix, Test_Weight_Matrix, Parameters)
        ALS_0.train()
        ALS_0.draw()
    else:
        Data=loadMovieLens(Parameters,dic_path)
        Models=[]
        for i in range(0,5):
            tic=time.time()
            print('Begin Trainning Model{}'.format(i))
            model=ALS(Data[0][i],Data[1][i],Data[2][i],Data[3][i],Parameters)
            model.train()
            print('Model_{} Completed. Time {:0.2f}s'.format(i,time.time()-tic))
            Models.append(model)
        dic={'Model_0':Models[0].test_Loss,
             'Model_1': Models[1].test_Loss,
             'Model_2': Models[2].test_Loss,
             'Model_3': Models[3].test_Loss,
             'Model_4': Models[4].test_Loss,}
        draw(Parameters['Max Loops'],dic,'loops','Loss','Test Loss for different Models')
