# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 18:29:17 2017

@author: ASUS
"""

import os
import numpy as np
import scipy.linalg as linalg
import cv2
import operator
from numpy import *
from PIL import Image  
from PIL import ImageOps
import numpy
from numpy import linalg
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pylab import *

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r

def myLDA(A,Labels):
    # function [W,m]=myLDA(A,Label)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # m: mean of each projection


    classLabels = np.unique(Labels)
    classNum = len(classLabels)
    dim,datanum = A.shape
    totalMean = np.mean(A,1)
    partition = [np.where(Labels==label)[0] for label in classLabels]
    classMean = [(np.mean(A[:,idx],1),len(idx)) for idx in partition]
    #compute the within-class scatter matrix
    W = np.zeros((dim,dim))
    for idx in partition:
        W += np.cov(A[:,idx],rowvar=1)*len(idx)

    #compute the between-class scatter matrix
    B = np.zeros((dim,dim))
    for mu,class_size in classMean:
        offset = mu - totalMean
        B += np.outer(offset,offset)*class_size

    #solve the generalized eigenvalue problem for discriminant directions
    W1=mat(W)
    W2=W1.I
    W3=np.array(W2)
    ew,ev = linalg.eig(W3.dot(B))
    
    sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
    selected_ind = [ind for ind,val in sorted_pairs[:classNum-1]]
    LDAW = ev[:,selected_ind]
    Centers = [np.dot(mu,LDAW) for mu,class_size in classMean]
    Centers = np.array(Centers).T
    return LDAW, Centers, classLabels



def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # LL: eigenvalues
    # m: mean of columns of A

    # Note: "lambda" is a Python reserved word


    # compute mean, and subtract mean from every column
    [r,c] = A.shape
    m = np.mean(A,1)
    A = A - np.tile(m, (c,1)).T
    B = np.dot(A.T, A)
    [d,v] = linalg.eig(B)

    # sort d in descending order
    order_index = np.argsort(d)
    order_index =  order_index[::-1]
    print(order_index)
    d = d[order_index]
    v = v[:, order_index]

    # compute eigenvectors of scatter matrix
    W = np.dot(A,v)
    Wnorm = ComputeNorm(W)

    W1 = np.tile(Wnorm, (r, 1))
    W2 = W / W1
    
    LL = d[0:-1]

    W = W2[:,0:-1]      #omit last column, which is the nullspace
    
    return W, LL, m

def read_faces(directory):
    # function faces = read_faces(directory)
    # Browse the directory, read image files and store faces in a matrix
    # faces: face matrix in which each colummn is a colummn vector for 1 face image
    # idLabels: corresponding ids for face matrix

    A = []  # A will store list of image vectors
    Label = [] # Label will store list of identity label
 
    # browsing the directory
    for f in os.listdir(directory):
        if not f[-3:] =='bmp':
            continue
        infile = os.path.join(directory, f)
        im = cv2.imread(infile, 0)
        # turn an array into vector
        im_vec = np.reshape(im, -1)
        A.append(im_vec)
        name = f.split('_')[0][-1]
        Label.append(int(name))

    faces = np.array(A, dtype=np.float32)
    faces = faces.T
    idLabel = np.array(Label)

    return faces,idLabel

def float2uint8(arr):
    mmin = arr.min()
    mmax = arr.max()
    arr = (arr-mmin)/(mmax-mmin)*255
    arr = np.uint8(arr)
    return arr



    import os
from numpy import *
faces,idLabel=read_faces("C:/Users/ASUS/Desktop/face/train")
W, LL, me=myPCA(faces)
K=30
We = W[:,: K]

classLabels = np.unique(idLabel)
classNum = len(classLabels)
partition = [np.where(idLabel==label)[0] for label in classLabels]

im0=mat(We.T)
facest=faces.T
X11=[]
im0=mat(We.T)
im2=mat(me)

K1=90
W1 = W[:,: K1]
A = []
im4=mat(W1.T)

for i in range(0,120):
    x=faces.T[i]
    im5=mat(x)
    im6=mat(me)
    x0=im4*(im5-im6).T
    im_vec = np.reshape(x0, -1)
    A.append(im_vec)
    
A1 = np.array(A, dtype=np.float32) 
A2=mat(A1)   
A2=np.array(A2)
LDAW, Centers, classLabels=myLDA(A2.T,idLabel)

def pcalda(alpha):
    for i in range(0,120):
        xpca=facest[i]
        im1=mat(xpca)
        ye = im0*(im1-im2).T
        
    
        x12=array(ye)
        x13=np.reshape(x12, -1)
        X11.append(x13)
    X12 = np.array(X11, dtype=np.float32)
    
    
    Z1 = [np.mean(X12.T[:,idx],1) for idx in partition]
    Zarray=array(Z1)
    Z=mat(Zarray)
    A=np.vstack((alpha*Z.T,(1-alpha)*Centers))
    Z=A   
    
    faces1,idLabel1=read_faces("C:/Users/ASUS/Desktop/face/test")
    faces1t=faces1.T
    confusion=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    for i in range(0,120):
        x=faces1t[i]
        im1=mat(x)
        ye = im0*(im1-im2).T
        
        x=faces1t[i]
        wft=mat(LDAW.T)
        w1t=mat(W1.T)
        minus=mat(x-me)
        y1=wft*w1t
        yf=y1*minus.T
        
        
        y=np.vstack((alpha*ye,(1-alpha)*yf))
        
        x12=array(y)
        x13=np.reshape(x12, -1)
        t1=x13-Z.T[0]
        minvec=t1*t1.T
        minnum=0
        for j in range(1,10):
            t1=x13-Z.T[j]
            t1dis=t1*t1.T
            temp=minvec
            minvec=min(minvec,t1dis)
            if minvec != temp:
                minnum=j
        confusion[idLabel1[i]][minnum]=confusion[idLabel1[i]][minnum]+1
    
    sumright=0
    for i in range(0,10):
        sumright=sumright+confusion[i][i]
    print("alpha=%.1f accuracy=%0.2f"%(alpha,(sumright/120*100)))
    point=[alpha,sumright/120*100]
    return point


alphav=[0.1*x for x in range(1,10)]
A=[]
for i in range(0,9):
    point=pcalda(alphav[i])
    A.append(point)

x=[]
y=[]
for i in range(0,9):
    x.append(A[i][0])
    y.append(A[i][1])
    

plt.plot(x,y,"b-.",linewidth=1)
plt.title('accuracy in different alpha')
plt.xlabel("alpha")
plt.ylabel("accuracy")
pyplot.savefig("Q5.png")