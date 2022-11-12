import numpy as np
from numpy import linalg as lin
from extract import *
from sympy import *

def mean(arr):
    result = [0] * 2048
    for i in range(len(arr)):
        result = np.add(result, arr[i])
    result = result / len(arr)
    return result

def selisih(arr, mean):
    for i in range(len(arr)):
        arr[i] = arr[i] - mean
    return arr

def covarian(mat):
    A= np.array(selisih(mat, mean(mat)))
    AT= A.transpose()
    covarian = np.matmul(A, AT)
    return covarian


#Mencari eigen value dan vector
def eigen_qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    
    return Q, A


#Buat cari vektor normalisasi
def make_householder(a):

    u = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    u[0] = 1
    H = np.eye(a.shape[0])

    H -= (2 / np.dot(u, u)) * u[:, None] @ u[None, :]
    return H    

def qr_iteration(A):
  
    #Algorithm to find eigenValues and eigenVector matrix using simultaneous power iteration.

    n, m = A.shape 
    Q = np.random.rand(n, m) #Make a random n x k matrix
    Q, _ = eigen_qr(Q) #Use QR decomposition to Q

 
    for i in range(100):
        Z = A.dot(Q)
        Q, R = eigen_qr(Z)
    #Do the same thing over and over until it converges
    return np.diag(R), Q



def eigenFace(selisih, vectorEigen):
    #cari matriks eigen face = selisih antar citra x vektor eigen
    #Kalo fotonya 4, eigen vectornya 4x4, matriks selsish 2048x4
    eFace = np.multiply(np.transpose(selisih), vectorEigen)
    
    #Kalo 4 image berarti selisihnya 2048 x 4 berarti vector eigennya 4x3
    wFace = np.multiply(np.transpose(eFace), selisih)
    
    return wFace



path = "src/dataset/pins_Adriana Lima"
extract = batch_extractor(path)
cov = covarian(batch_extractor(path))
matSelisih = selisih(extract, mean(extract))
eigVal, eigVec = qr_iteration(cov)
face = eigenFace(matSelisih ,eigVec)
print(eigVec)
print("\n\n\n")
print(face)







