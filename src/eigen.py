import numpy as np
from numpy import linalg as lin

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

def conj_transpose(a): 
    return np.conj(np.transpose(a))

def proj_ab(a,b):                  
    return np.dot(a, conj_transpose(b)) / lin.norm(b) * b

# Mencari eigen menggunakan Gram-Schmidt Orthogonalization
def qr_gs(A, type= np.float32):    
    A = np.array(A, dtype=type)
    (m, n) = np.shape(A)
    
    (m, n) = (m, n) if m > n else (m, m)

    Q = np.zeros((m, n), dtype=type)
    R = np.zeros((m, n), dtype=type)
    
    for k in range(n):       
        pr_sum = np.zeros(m, dtype=type)
        
        for j in range(k):
            pr_sum += proj_ab(A[:,k], Q[:,j])
            
        Q[:,k] = A[:,k] - pr_sum                                             
        Q[:,k] = Q[:,k] / lin.norm(Q[:,k])

    if type == complex:        
        R = conj_transpose(Q).dot(A)
    else: R = np.transpose(Q).dot(A)
        
    return -Q, -R


