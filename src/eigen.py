import numpy as np
from numpy import linalg as lin
from extract import *
from sympy import *
import collections


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

 
    for i in range(1000):
        Z = A.dot(Q)
        Q, R = eigen_qr(Z)
    #Do the same thing over and over until it converges
    return np.diag(R), Q

'''def bestFace(eigVal, eigVec):
    bestValue = len(eigVal) // 2
    result = {}
    for i in range(len(eigVal)):
        inserteigVal = eigVal[i]
        result[inserteigVal] = eigVec[i]
    
    new = dict(collections.OrderedDict(sorted(result.items())))
    bestFace = dict(list(new.items())[bestValue: len(eigVal)])
    
    besteigVal = []
    besteigVector = []
    
    for items in bestFace:
        besteigVal.append(items)
        change = np.ndarray.tolist(bestFace[items])
        besteigVector.append(change)
        
    return besteigVal, besteigVector'''
        

def eigenFace(vectorEigen, selisih):
    #cari matriks eigen face = selisih antar citra x vektor eigen
    #Kalo fotonya 4, eigen vectornya 4x4, matriks selisih 2048x4
    eFace = np.matmul(vectorEigen, selisih)
    return eFace


def weightFace(eigFace,selisih):
    #Kalo 4 image berarti selisihnya 2048 x 4 berarti vector eigennya 4x3
    wFace = np.matmul(np.transpose(eigFace), np.transpose(selisih))
    return wFace

def euclideanDistance(weight, queryWeight):
    distance = np.linalg.norm(weight - queryWeight, axis = 0)
    sortDistance = np.sort(distance)
    
    return sortDistance

def threshold(distance):
    max = np.argmax(distance)
    threshold = 0.5 * (np.sqrt(max))
    
    return threshold

#eigenvalue paling gede, ambil minum banyak dataset
path = "D:/00_STEI ITB/03_SMT3/Aljabar Geometri dan Linear/TUBES 2/testdata"
names, extract = batch_extractor(path) #nx2048

cov = covarian(extract)
matSelisih = selisih(extract, mean(extract))
eigVal, eigVec = qr_iteration(cov)
print(eigVec)
print("\n")
eiglibval, eiglibvec = np.linalg.eig(cov)
print(eiglibvec)
print(tes.shape)
bestVal, bestVec = bestFace(eigVal, eigVec) #bestVec (n/2) x n, #matSelisih nx2048
face = eigenFace(eigVec, matSelisih)
bestface = eigenFace(bestVec, matSelisih)
#tes3 = np.array(bestface)#(n/2) x 2048
weight = weightFace(bestface, matSelisih) 
'''tes = np.array(matSelisih)
print(tes.shape) #11x2048
tes2 = np.array(bestVec)
print(tes2.shape) #6x11
tes3 = np.array(bestface)
tes4 = np.array(weight)
print(tes3.shape) #6x2048
print(tes4.shape) #6x11
#tes = np.array(weight)
#print(tes.shape)'''

path1 = "D:/00_STEI ITB/03_SMT3/Aljabar Geometri dan Linear/TUBES 2/testdata/Alex Lawther3_104.jpg"
extracttes = extract_features(path1) #1x2048
#print(extracttes)
extract1 = []
extract1.append(extracttes)
query = extract1

matSelisih1 = selisih(query, mean(extract)) #1x2048 query-mean
changeSelisih = np.array(matSelisih1)
tSelisih = np.transpose(changeSelisih)
#eigenface 2048xn
queryWeight = np.matmul(face, np.transpose(matSelisih1)) #6x2048 x 2048x1
queryWeightBest = np.matmul(bestface, np.transpose(matSelisih1))
distance = np.linalg.norm(weight - queryWeight, axis = 0) #7x7 
distanceBest = np.linalg.norm(weight - queryWeightBest, axis = 0) #7x7 
#bestMatch = np.argmin(distance)
print(distance)
print(distanceBest)
#print(names[bestMatch])



#TAKBIRRRRRR

#eigedistance semua
#eigendistance bestface
