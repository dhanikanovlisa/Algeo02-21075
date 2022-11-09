import cv2
import numpy as np
import os
from matplotlib.pyplot import imread
import sys
np.set_printoptions(threshold=sys.maxsize)

# Feature extractor
def extract_features(image_path, vector_size=32):
    image = imread(image_path)
    # image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
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
        i += 1 
    result = list(result.values())
    return result

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

# path0 = 'src\dataset\pins_Adriana Lima\Adriana Lima0_0.jpg'
# path1 = 'src\dataset\pins_Adriana Lima\Adriana Lima1_1.jpg'
# path = 'src\dataset\pins_Adriana Lima'

# res = batch_extractor(path)
# mean_res = mean(res)

# A= np.array(selisih(res, mean_res))
# AT= A.transpose()
# print("yeah")
# covarian = np.matmul(A, AT)
# print(covarian.shape)
# print(covarian)

