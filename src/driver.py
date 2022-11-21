from extract import *
from eigen import *
from faceDetection import *

dataPath = "src/dataset/dataset1"

names, extract = batch_extractor(path) #nx2048


matriksKovarian = covarian(extract)
matriksSelisih = selisih(extract, mean(extract))
eigValue, eigVector = qr_iteration(cov)
eigenFace = eigenFace(matriksSelisih, eigVector)
weight = weightFace(eigenFace, matriksSelisih)

imagePath = "src/dataset/dataset1/Adriana Lima10_2.jpg"
query = extractFace(imagePath, extract)
queryWeight = queryWeight(eigenFace, query)
eucDistance = euclideanDistance(weight, queryWeight)
minDistance, match = bestMatch(names, eucDistance)
print(minDistance)
print(match)

