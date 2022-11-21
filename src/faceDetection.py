from extract import *
from eigen import *

def extractFace(imagePath, extractData):
    extractMatrix = extract_features(imagePath)
    extract = []
    extract.append(extractMatrix)
    
    query = extract
    querySelisih = selisih(query, mean(extractData))
    
    return querySelisih

def queryWeight(eigenFace, querySelisih):
    queryWeight = np.matmul(np.transpose(eigenFace), np.transpose(querySelisih))
    
    return queryWeight

def euclideanDistance(weight, queryWeight):
    #weight nxn
    #queryWeight nx1
    weightTranspose = np.transpose(weight) #nxn
    queryWeightTranspose = np.transpose(queryWeight) #1xn
    
    
    substract = np.subtract(queryWeightTranspose, weightTranspose)
    getRow = np.array(substract)
    row = getRow.shape[0]

    distance = []
    for i in range(0, row):
        sum = 0
        for j in range(0, row):
            sum += ((substract[i][j] ** 2))
            akar = math.sqrt(sum)
        distance.append(akar)
    
    return distance

def bestMatch(names, distance):
    
    min = distance[0]
    size = len(distance)
    
    for i in range(0, size):
        if (min > distance[i]):
            min = distance[i]
    
    hook = 0
    for j in range(0, size):
        if (min == distance[j]):
            hook = j
            
    faceMatch = names[hook]
            
    return min, faceMatch



