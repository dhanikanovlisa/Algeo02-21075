import numpy as np

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
    trans_arr = np.transpose(arr)
    covarian = np.matmul(arr, trans_arr)
    return covarian