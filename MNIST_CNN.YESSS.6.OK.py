import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('mnist_test_2k.csv',sep=',',header=1)
xx=np.transpose(df)
x=np.array(xx[:785])
y=x[0]
y=np.array([y[i:i+1] for i in range(0,len(y),1)])
a=np.delete(x,(0),axis=0)
X=a.T
ax=[X[0][i:i+28] for i in range(0,len(X[0]),28)]
# INITIALIZE RANDOM WEIGHTS
syn0 = 2*np.random.random((784,1000)) - 1
syn0.shape
syn1 = 2*np.random.random((1000,1)) - 1
def xrange(x):
    return iter(range(x))
# BACKPROPAGATION
for j in xrange(2000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
# SIGMOID FUNCTION
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
np.mean(nonlin(syn1))
#### ALL WEIGHTS FROM REGRESSION OK
c=nonlin(syn0)
# ARG MAX SIGMOID
f=[i for i,x in enumerate(c[0]) if x == max(c[0])]
f=[int(i) for i in f]
ax2=[X[f][0][i:i+28] for i in range(0,len(X[f][0]),28)]
df=pd.read_csv('mnist_train_100.csv',sep=',',header=1)
xx=np.transpose(df)
x=np.array(xx[:785])
y2=x[0]
a=np.delete(x,(0),axis=0)
X1=a.T
X1.shape
s=syn0.T
s.shape
X1[7]

ss=[]
for i in range(0,999):
    ss.append(np.dot(s[i],X1[0]))
abs(np.array(ss))
t=[i for i,x in enumerate(abs(np.array(ss))) if x == max(abs(np.array(ss)))]
X[t[0]]

plt.subplot(221)
plt.imshow(ax2, cmap=plt.get_cmap('gray'))
plt.title('MNIST Train Set')
ax3=[X[t[0]][i:i+28] for i in range(0,len(X[t[0]]),28)]
ax3
plt.imshow(ax3, cmap=plt.get_cmap('gray'))
plt.title('MNIST Test Set')
np.mean(nonlin(syn1))



