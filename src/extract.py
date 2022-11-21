import cv2
import numpy as np
import os
from matplotlib.pyplot import imread
from numpy.linalg import eig

# Feature extractor
def extract_features(image_path, vector_size=32):
    image = imread(image_path)
    try:
        alg = cv2.KAZE_create()
        kps = alg.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()
        
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
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


