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
    return Q, -A


#Buat cari vektor normalisasi
def make_householder(a):

    u = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    u[0] = 1
    H = np.eye(a.shape[0])

    H -= (2 / np.dot(u, u)) * u[:, None] @ u[None, :]
    return H    

def eigen_vektor():
    #Ambil nilai eigen
    path = "C:/Users/dhani/Algeo02-21075/src/dataset/pins_Alexandra Daddario"
    A = covarian(batch_extractor(path))
    Q, R = eigen_qr(A)
    
    eigenValue = np.diag(R.round(5)) #isinya eigenvalue
    m = np.size(A, 1)
    matrixIdentitas = np.eye(m) #matriks identita
    matrixParametrik = np.zeros((m , m))
    eigenVec = np.zeros((m, m))

    for i in eigenValue:
        if i != 0:
            matrixParametrik = np.subtract(np.multiply(matrixIdentitas, i), A)
            m = Matrix(matrixParametrik)
            print(m.nullspace())

eigen_vektor()






