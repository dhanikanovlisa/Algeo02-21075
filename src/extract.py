import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import sys
np.set_printoptions(threshold=sys.maxsize)

# Feature extractor
def extract_features(image_path, vector_size=32):
    image = imread(image_path)
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
    i = 0
    result = {}
    for f in files:
        result[i] = extract_features(f)
        i+=1
    return result

# path2 = 'src\dataset\pins_Adriana Lima\Adriana Lima1_1.jpg'
# path = 'src\dataset\pins_Adriana Lima'

# res = batch_extractor(path)
# res2 = extract_features(path2)
# print(res[1])
# print(res2)
