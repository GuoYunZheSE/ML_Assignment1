import numpy
from numpy import random
#Cost Function:0.5*lambda*W'*W+0.5*(Y-X*W)'*(Y-X*W)

#Read Data
m=input("Please input the Columns:\n")
n=input("Please input the Columns:\n")
m=int(m)
n=int(n)
Data=numpy.loadtxt('C:/Users/user/documents/visual studio 2017/Projects/ML_Assignment1/ML_Assignment1/DataSet/abalone.txt')
Y=Data[:,0].reshape(m,1)
X=Data[:,1:n+1]

#initial W, lambda and our learning rate N
W=numpy.random.random(size=(n,1))
m_lambda=0.01
N=0.00007

#Compute Grad http://blog.sina.com.cn/s/blog_6cb263210101csq0.html
def Grad():
    grad=0.5*m_lambda*W+(numpy.dot(-X.transpose(),Y)+numpy.dot(X.transpose(),numpy.dot(X,W)))
    return grad

#Compute Closed_Form Solution, we want grad to be 0, therefore W=(0.5*lambda+X'X)^(-1)*X'Y
W_CFS=(numpy.dot(X.transpose(),X)+0.5*m_lambda)     #(0.5*lambda+X'X)
W_CFS=numpy.linalg.inv(W_CFS)                       #(0.5*lambda+X'X)^(-1)
W_CFS=numpy.dot(W_CFS,numpy.dot(X.transpose(),Y))   #(0.5*lambda+X'X)^(-1)*X'Y
print(W_CFS)

#BGD
max_loop=100000#in case it won't converage
epsilon=0.00000001
count=0
error=numpy.zeros( (n,1) )
finish=0
while count<=max_loop:
    count+=1
    W=W-N*Grad()
    if(numpy.linalg.norm(W-error)<epsilon):
        finish=1
        break
    else:
        error=W
        print(count)
        print(W)
#Compare
print("BGD:",W)
print("CFS:",W_CFS)
'''
BGD: [[ -0.09032331]
 [  7.19288428]
 [ 12.85533016]
 [ 14.59087635]
 [  8.73965705]
 [-21.21042226]
 [-11.93278079]
 [  6.46321643]]
CFS: [[ -0.09032432]
 [  7.14952992]
 [ 12.91605734]
 [ 14.58911064]
 [  8.77825922]
 [-21.24525835]
 [-11.9898167 ]
 [  6.41726054]]
 '''