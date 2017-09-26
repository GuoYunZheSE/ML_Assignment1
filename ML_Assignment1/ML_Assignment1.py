#201530611524 郭蕴喆 软件工程三班
import numpy
from numpy import random
#Cost Function:0.5*lambda*W'*W+0.5*(Y-X*W)'*(Y-X*W)

#Read Data
m=input("Please input the Columns:\n")#m in my datadset is 4177
n=input("Please input the Columns:\n")#n in my datadset is 8
m=int(m)
n=int(n)
Data=numpy.loadtxt('./DataSet/abalone.txt')
Y=Data[:,0].reshape(m,1)
X=Data[:,1:n+1]

#initial W, lambda and our learning rate N
W=numpy.random.random(size=(n,1))
m_lambda=0.01
N=0.000085

#Compute Grad 
def Grad():
    grad=m_lambda*W+(numpy.dot(X.transpose(),numpy.dot(X,W)))
    return grad

#Compute Closed_Form Solution, we want grad to be 0, therefore W=(0.5*lambda+X'X)^(-1)*X'Y
W_CFS=(numpy.dot(X.transpose(),X)+m_lambda)     #(lambda+X'X)
W_CFS=numpy.linalg.inv(W_CFS)                       #(lambda+X'X)^(-1)
W_CFS=numpy.dot(W_CFS,numpy.dot(X.transpose(),Y))   #(lambda+X'X)^(-1)*X'Y


#BGD
max_loop=100000#in case it won't converage
epsilon=0.000001
count=0
error=numpy.zeros((n,1))
finish=0
Tensor=numpy.dot(-X.transpose(),Y);#It never change during our process, so I put it in a tensor
while count<=max_loop:
    count+=1
    W=W-N*(Grad()+Tensor)
    if(numpy.linalg.norm(W-error)<epsilon):
        finish=1
        break
    else:
        error=W
        #print(count)  #You can choose whether to print Count/W 
        #print(W)
   
#Compare
W_Original=numpy.dot(X.transpose(),X)
W_Original=numpy.linalg.inv(W_Original)
W_Original=numpy.dot(numpy.dot(W_Original,X.transpose()),Y)

print("参数:","m=",m,"n=",n)
print("BGD:",W)
print("CFS:",W_CFS)
print("Original:",W_Original)

'''
测试结果：
参数: m= 4177 n= 8
BGD: [[ -0.09026013]
 [  7.17334751]
 [ 12.88101512]
 [ 14.58984364]
 [  8.73951467]
 [-21.21019105]
 [-11.93193225]
 [  6.46178502]]
CFS: [[ -0.09032432]
 [  7.14952992]
 [ 12.91605734]
 [ 14.58911064]
 [  8.77825922]
 [-21.24525835]
 [-11.9898167 ]
 [  6.41726054]]
Original: [[ -0.09033082]
 [  7.1329695 ]
 [ 12.92884736]
 [ 14.61503969]
 [  8.7573378 ]
 [-21.22765678]
 [-11.96028156]
 [  6.44057868]]
 '''