import cv2
import numpy as np
import os
from matplotlib.pyplot import imread
import sys
from numpy.linalg import eig
import pickle
import csv

# Feature extractor
def extract_features(image_path, vector_size=32):
    imageRead = imread(image_path)
    image = cv2.resize(imageRead, (256,256), interpolation=cv2.INTER_AREA)
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print ('Error: ', e)
        return None

    return dsc

def batch_extractor(images_path):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    result = {}

    for f in files:
        name = os.path.basename(f)
        result[name] = extract_features(f)
    
    names = []
    extract = []
    
    for items in result:
        names.append(items)
        change = np.ndarray.tolist(result[items])
        extract.append(change)
        
            
    return names, extract

path = "D:/00_STEI ITB/03_SMT3/Aljabar Geometri dan Linear/TUBES 2/testdata"
names, extract = batch_extractor(path)
tes = np.array(extract)
print(tes.shape)