import pandas as pd
import numpy as np
import math
import random

m = 5 #number of samplers
k = 10 #grid lentgh
sigma1 = 0.5 #case a sigma
sigma2 = 0.8 #case b sigma
sigma3 = 4 #case b sigma
letters = ['A','C','K','T','W']
#Q1

def PrintGrid(ArrayIn):
    for i in range(ArrayIn.shape[0]):
        print("")
        for j in range(ArrayIn.shape[1]):
            if ArrayIn[i][j] == 1:
                print(" ", end = '')
            else:
                print("\u2588", end = '')
                
    print("\n")
    

G = [] #Letter grids
G.append(np.array( #Grid for letter A
                [[+1,+1,+1,-1,-1,-1,-1,+1,+1,+1],
                 [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
                 [+1,-1,-1,-1,+1,+1,-1,-1,-1,+1],
                 [+1,-1,-1,-1,+1,+1,-1,-1,-1,+1],
                 [+1,-1,-1,-1,+1,+1,-1,-1,-1,+1],
                 [+1,-1,-1,-1,-1,-1,-1,-1,-1,+1],
                 [+1,-1,-1,-1,-1,-1,-1,-1,-1,+1],
                 [+1,-1,-1,-1,+1,+1,-1,-1,-1,+1],
                 [+1,-1,-1,-1,+1,+1,-1,-1,-1,+1],
                 [+1,-1,-1,-1,+1,+1,-1,-1,-1,+1]]
                ))

G.append(np.array( #Grid for letter C
                [[+1,+1,+1,-1,-1,-1,-1,-1,-1,+1],
                 [+1,+1,-1,-1,-1,-1,-1,-1,-1,+1],
                 [+1,-1,-1,-1,+1,+1,+1,+1,+1,+1],
                 [+1,-1,-1,-1,+1,+1,+1,+1,+1,+1],
                 [+1,-1,-1,-1,+1,+1,+1,+1,+1,+1],
                 [+1,-1,-1,-1,+1,+1,+1,+1,+1,+1],
                 [+1,-1,-1,-1,+1,+1,+1,+1,+1,+1],
                 [+1,-1,-1,-1,+1,+1,+1,+1,+1,+1],
                 [+1,+1,-1,-1,-1,-1,-1,-1,-1,+1],
                 [+1,+1,+1,-1,-1,-1,-1,-1,-1,+1]]
                ))

G.append(np.array( #Grid for letter K
                [[-1,-1,-1,+1,+1,+1,-1,-1,-1,+1],
                 [-1,-1,-1,+1,+1,-1,-1,-1,+1,+1],
                 [-1,-1,-1,+1,-1,-1,-1,+1,+1,+1],
                 [-1,-1,-1,-1,-1,-1,+1,+1,+1,+1],
                 [-1,-1,-1,-1,-1,+1,+1,+1,+1,+1],
                 [-1,-1,-1,-1,-1,+1,+1,+1,+1,+1],
                 [-1,-1,-1,-1,-1,-1,+1,+1,+1,+1],
                 [-1,-1,-1,+1,-1,-1,-1,+1,+1,+1],
                 [-1,-1,-1,+1,+1,-1,-1,-1,+1,+1],
                 [-1,-1,-1,+1,+1,+1,-1,-1,-1,+1]]
                ))

G.append(np.array( #Grid for letter T
                [[-1,-1,-1,-1,-1,-1,-1,-1,-1,+1],
                 [-1,-1,-1,-1,-1,-1,-1,-1,-1,+1],
                 [+1,+1,+1,-1,-1,-1,+1,+1,+1,+1],
                 [+1,+1,+1,-1,-1,-1,+1,+1,+1,+1],
                 [+1,+1,+1,-1,-1,-1,+1,+1,+1,+1],
                 [+1,+1,+1,-1,-1,-1,+1,+1,+1,+1],
                 [+1,+1,+1,-1,-1,-1,+1,+1,+1,+1],
                 [+1,+1,+1,-1,-1,-1,+1,+1,+1,+1],
                 [+1,+1,+1,-1,-1,-1,+1,+1,+1,+1],
                 [+1,+1,+1,-1,-1,-1,+1,+1,+1,+1]]
                ))

G.append(np.array( #Grid for letter W
                [[-1,-1,-1,+1,-1,-1,+1,-1,-1,-1],
                 [-1,-1,-1,+1,-1,-1,+1,-1,-1,-1],
                 [-1,-1,-1,+1,-1,-1,+1,-1,-1,-1],
                 [-1,-1,-1,+1,-1,-1,+1,-1,-1,-1],
                 [-1,-1,-1,+1,-1,-1,+1,-1,-1,-1],
                 [-1,-1,-1,+1,-1,-1,+1,-1,-1,-1],
                 [-1,-1,-1,+1,-1,-1,+1,-1,-1,-1],
                 [+1,-1,-1,-1,-1,-1,-1,-1,-1,+1],
                 [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
                 [+1,+1,+1,-1,+1,+1,-1,+1,+1,+1]]
                ))
print("Initial patterns")
for i in range(m):
    PrintGrid(G[i])
    print("\n\n")

#Q2

X = [] #Letter vectors

for i in range(m):
    X.append(G[i].reshape(k*k,1))
  
#Q3

W = np.ndarray(buffer = np.empty((k,k)),shape=(k,k)) #Weight matrix initialized
W[:] = np.NaN

XoutPr =[] #Out products of vectors are calculated to find all Xi**Xj's
for i in range(m):
    XoutPr.append(np.matmul(X[i],X[i].T))
    
W = np.array(sum(XoutPr))
np.fill_diagonal(W,0)

#Hopfield Algorithm

def HopfieldAlgorithm(VectorIn,WeightIn):
    iteration = 0
    delta = 1
    Mu = VectorIn
    while delta != 0:
        print("\nIteration %d" % (iteration))
        print("Delta: {}".format(delta),end='')
        PrintGrid(Mu.reshape(k,k))
        MuOld = Mu
        Mu = np.matmul(WeightIn,MuOld)/abs(np.matmul(WeightIn,MuOld))
        delta = sum(sum(abs(Mu - MuOld)))
        iteration = iteration+1

    print("Delta: {}".format(delta),end='')
    return Mu

#Q4a
    
print("\n\nCase A:")
XdistA = []
for i in range(m):
    XdistA.append(X[i]+ np.ndarray(buffer = np.random.normal(0,sigma1,k*k),shape = (k*k,1)))
    XdistA[i] = XdistA[i]/abs(XdistA[i])

for i in range(m):
    print("\n\nLetter {}".format(letters[i]))
    HopfieldAlgorithm(XdistA[i],W)
       
#Q4b

print("\n\nCase B:")
XdistB = []
for i in range(m):
    XdistB.append(X[i]+ np.ndarray(buffer = np.random.normal(0,sigma2,k*k),shape = (k*k,1)))
    XdistB[i] = XdistB[i]/abs(XdistB[i])

for i in range(m):
    print("\n\nLetter {}".format(letters[i]))
    HopfieldAlgorithm(XdistB[i],W)
    
#Q4c

print("\n\nCase C:")
XdistC = []
for i in range(m):
    XdistC.append(X[i]+ np.ndarray(buffer = np.random.normal(0,sigma3,k*k),shape = (k*k,1)))
    XdistC[i] = XdistC[i]/abs(XdistC[i])

for i in range(m):
    print("\n\nLetter {}".format(letters[i]))
    HopfieldAlgorithm(XdistC[i],W)
        
