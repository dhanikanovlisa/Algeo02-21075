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

def eigen_vektor():
    #Ambil nilai eigen
    path = "src/dataset/pins_Adriana Lima"
    A = covarian(batch_extractor(path))
    Q, R = np.linalg.qr(A)
    W, V = lin.eig(A)
    # print(R)

    print("Eigen Lib: ", W)
    
    eigenValue, yea = qr_iteration(R) #isinya eigenvalue
    print("Eigen Value: ", eigenValue)
    m = np.size(A, 1)
    matrixIdentitas = np.eye(m) #matriks identitas
    matrixParametrik = np.zeros((m , m))
    eigenVec = np.zeros((m, m))

    for i in eigenValue:
        if i != 0:
            matrixParametrik = np.subtract(np.multiply(matrixIdentitas, i), A)
            m = Matrix(matrixParametrik)
            # print(m.nullspace())

# eigen_vektor()

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


path = "src/dataset/pins_Adriana Lima"
cov = covarian(batch_extractor(path))

W, V = lin.eig(cov)
W1, V1 = qr_iteration(cov)

print(type(W))
W = np.sort(W)[::-1]
W1 = np.sort(W1)[::-1]


print("EigenValue Lib: \n", W,"\n\n")
print("EigenValue code: \n", W1,"\n\n")
print("EigenVector Lib: \n", V,"\n\n")







