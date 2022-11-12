import numpy as np
from numpy import linalg as lin
from extract import *
from sympy import *

def euclideanDistance(eigenVec, testFace, eigenFace):
    new = np.multiply(eigenVec, testFace)
    row = eigenFace.shape[0]
    for i in range(0, row):
        distanceMatrix = np.substract(new, eigenFace[i])
        
    