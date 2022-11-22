from extract import *
from eigen import *
from faceDetection import *

'''dataPath = "src/dataset/dataset1"

names, extract = batch_extractor(dataPath) #nx2048
matriksKovarian = covarian(extract) #nxn
matriksSelisih = selisih(extract, mean(extract)) #nx2048
eigValue, eigVector = getEigen(matriksKovarian) #nxn
eigenFace = eigenFace(matriksSelisih, eigVector) #nxn
weight = weightFace(eigenFace, matriksSelisih) #nxn 

imagePath = "src/dataset/dataset1/Anne Hathaway12_313.jpg"
query = extractFace(imagePath, extract)
queryWeight = queryWeight(eigenFace, query)
eucDistance = euclideanDistance(weight, queryWeight)
minDistance, match = bestMatch(names, eucDistance)
print(minDistance)
print(match)
'''
