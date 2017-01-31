import cv2
import numpy as np
import pandas as pd
from scipy.cluster.vq import *


def SIFT_extractor(image_paths):
    detector = cv2.xfeatures2d.SIFT_create()

    des_list = []

    print('Total: ', len(image_paths))
    count=0

    print('----')
    for image_path in image_paths:
        im = cv2.imread(image_path,0)
        im = cv2.resize(im, (400, 250))
        kpts, des = detector.detectAndCompute(im, None)
        des_list.append((image_path, des))
        if count%(int(len(image_paths)/10))==0:
            print(round(count/len(image_paths)*100), '%')
        count=count+1
    print('----')
    return des_list

#des_list=SIFT_extractor(image_paths)

def transform_descriptors(des_list,image_classes):
    descriptors=[]
    labels=[]
    count=1
    for dd in des_list[1:]:
        l=image_classes[count]
        count=count+1
        for d in dd[1]:
            descriptors.append(d)
            labels.append(l)

    descriptors=np.array(descriptors)
    return  descriptors, labels


#descriptors, labels=transform_descriptors(des_list,image_classes)
