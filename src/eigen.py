import numpy as np
from numpy import linalg as lin
from extract import *

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
    path = "C:/Users/dhani/Algeo02-21075/src/dataset/pins_Adriana Lima"
    A = covarian(batch_extractor(path))
    Q, R = eigen_qr(A)
    
    eigenValue = np.diag(R.round(5)) #isinya eigenvalue
    m = np.size(A, 1)
    matrixIdentitas = np.eye(m) #matriks identita
    eV = np.zeros((m , m))
    result = np.zeros((m , m))
    matrixParameterik = np.zeros((m , m))

    for i in eigenValue:
        b = np.zeros((m, 1))
        if i != 0:
            matrixParameterik = np.subtract(np.multiply(matrixIdentitas, i), A)
            solve = np.linalg.solve(matrixParameterik, b)
            print(solve)

path = "C:/Users/dhani/Algeo02-21075/src/dataset/pins_Adriana Lima"
A = covarian(batch_extractor(path))
R = np.linalg.qr(A)[1]
w,v = np.linalg.eig(R)
print(v)

    








