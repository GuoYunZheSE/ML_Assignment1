import numpy
from numpy import random
#Cost Function:0.5*lambda*W'*W+0.5*(Y-X*W)'*(Y-X*W)
Data=numpy.loadtxt('C:/Users/user/documents/visual studio 2017/Projects/ML_Assignment1/ML_Assignment1/DataSet/abalone.txt')
Y=Data[:,0]
X=Data[:,1:9]

#initial W, lambda and our learning rate N
W=numpy.random.random(size=(8,1))
m_lambda=0.01
N=0.001

#Compute Grad
grad=0.5*(W.transpose()+W)+(numpy.dot(-X.transpose(),Y)+numpy.dot(X.transpose(),numpy.dot(X,W)))

#Compute Closed_Form Solution
W_CFS=(numpy.dot(X.transpose(),X))
W_CFS=numpy.linalg.inv(W_CFS)
W_CFS=numpy.dot(W_CFS,X.transpose())
W_CFS=numpy.dot(W_CFS,Y)
print(W_CFS)

#BGD http://www.cnblogs.com/21207-iHome/p/5222993.html
for i in (1:4177):
