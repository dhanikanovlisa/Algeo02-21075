import numpy as np
import extract

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

def kovarian(arr):
    compute = selisih(arr, mean(arr))
    trans = np.transpose(compute)
    covarian = np.matmul(compute, trans)
    return covarian

def eigen_qr(arr, iterations=10000):
    Ak = np.copy(arr)
    n = Ak.shape[0]
    QQ = np.eye(n)
    for k in range(iterations):
        s = Ak.item(n-1, n-1)
        smult = s * np.eye(n)
        Q, R = np.linalg.qr(np.subtract(Ak, smult))
        Ak = np.add(R @ Q, smult)
        QQ = QQ @ Q
    return Ak, QQ


# Testing
"""
path = 'src\dataset\pins_Adriana Lima'

vec = extract.batch_extractor(path)
kov = kovarian(vec)
print(np.shape(kov))
eigval, eigvec = np.linalg.eig(kov)
eigqr, eigvecqr = eigen_qr(kov)

print(eigvecqr, "\n\n")
print(eigvec)
"""