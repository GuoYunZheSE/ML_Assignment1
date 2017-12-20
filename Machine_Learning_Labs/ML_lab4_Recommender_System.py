import numpy as np
import sklearn
import matplotlib
import time


def loadMovieLensTrain(parametes,directory_path):
    rows=parametes['Row']
    cols=parametes['Cols']
    mode=parametes['Mode']
    # Method:Hold-Out
    if mode==0:
        file_path=directory_path+'/u.data'
        Rating_Matrix=np.zeros(shape=(rows,cols))
        origin_data=np.loadtxt(file_path)
        for i in range(0,origin_data.shape[0]):
            user_id=int(origin_data[i][0])
            item_id=int(origin_data[i][1])
            rating=origin_data[i][2]
            Rating_Matrix[user_id-1,item_id-1]=rating
        return Rating_Matrix


def loss(R,P,Q,m_lambda):
    Predict=P.dot(Q)
    m=R.shape[0]
    n=R.shape[1]
    k=P.shape[1]
    Loss=0
    '''
    Regulization=0
    SEL=((R-Predict)**2).sum()
    RP=P.dot(P.transpose())
    RQ=Q.transpose().dot(Q)
    for i in range(0,RP.shape[0]-1):
        Regulization=Regulization+m_lambda*np.square(RP[i][i])
    for i in range(0,RQ.shape[0]-1):
        Regulization=Regulization+m_lambda*np.square(RP[i][i])
    Loss=SEL+Regulization
    '''
    for u in range(0,m-1):
        for i in range(0,n-1):
            if R[u,i]==0:
                continue
            else:
                Npu=np.count_nonzero(P,1)[u]
                Nqi=np.count_nonzero(Q,0)[i]
                P_temp=P[u,:].reshape(1,k)
                Q_temp=Q[:,i].reshape(k,1)
                Loss=Loss+np.square(R[u,i]-Predict[u,i])+m_lambda*(Npu*P_temp.dot(P_temp.transpose)+
                                                             Nqi*Q_temp.dot(Nqi*Q_temp.transpose()))
    return Loss


def ALS(R,P,Q,Parameters):
    tic=time.time()
    count = 1
    max_loops=Parameters['Max Loops']
    rows=Parameters['Row']
    cols=Parameters['Cols']
    k=Parameters['K']
    m_lambda=Parameters['lambda']
    while count <= max_loops:
        rmse=1
        count=count+1
        if rmse<=Parameters['epsilon']:
            print('ALS Completed in loops {}, Time:{:0.2f}'.format(count,time.time()-tic))
            break
        else:
            for u in range(0, rows - 1):
                Caculus = 0
                QRu = R[u, :].reshape(1,cols).dot(Q.transpose())
                for i in range(0, cols - 1):
                    if R[u, i] == 0:
                        continue
                    else:
                        Npu = np.count_nonzero(P, 1)[u]
                        Vector_Q = Q[:, i].reshape(k, 1)
                        Caculus = Caculus + Vector_Q.dot(Vector_Q.transpose()) + m_lambda * Npu * np.ones(shape=(k, k))
                P[u, :] = QRu.dot(np.linalg.inv(Caculus))

            for i in range(0, cols - 1):
                Caculus = 0
                Vector_Q = R[:, i].reshape(1, rows)
                PRi = P.transpose() .dot( Vector_Q.transpose())
                for u in range(0, rows - 1):
                    if R[u, i] == 0:
                        continue
                    else:
                        Nqi = np.count_nonzero(Q, 0)[i]
                        Caculus = Caculus + (P[u, :].reshape(1,k)).transpose().dot((P[u, :].reshape(1,k))) \
                                  + m_lambda * Nqi * np.ones(shape=(k, k))
                Q[:, i] = ((np.linalg.inv(Caculus).dot(PRi))).reshape(k,)
            rmse=RMSE(R,P,Q)
        print('Loops:{}, RMSE:{:0.5f}'.format(count,rmse))
    print('ALS Completed, Time:{:0.2f}'.format(count, time.time() - tic))



def RMSE(R,P,Q):
    Predict = P.dot(Q)
    SEL=((R-Predict)**2).sum()
    return np.sqrt(SEL/(R.shape[0]*R.shape[1]))

if __name__ == '__main__':
    dic_path='/home/lucas/Codes/GitHub/ML_Assignment1/ML_Assignment1/DataSet/ml-100k'
    Parameters={
        'Row':943,
        'Cols':1682,
        'Mode':0,
        'K':4,
        'lambda':0.01,
        'Max Loops':10000,
        'epsilon':0.00001
    }
    Rating_Matrix = loadMovieLensTrain(Parameters,dic_path)
    P_Matrix=np.random.random(size=(Parameters['Row'],Parameters['K']))
    Q_Matrix=np.random.random(size=(Parameters['K'],Parameters['Cols']))
    ALS(Rating_Matrix,P_Matrix,Q_Matrix,Parameters)
