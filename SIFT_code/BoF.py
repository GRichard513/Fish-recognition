import numpy as np
import pandas as pd
from sklearn import cluster
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler


# Perform k-means clustering
from sklearn.cluster import KMeans

def create_voc(descriptors,k=100):
    km=KMeans(k)
    #km.fit(descriptors)
    voc,_=kmeans(descriptors,k)
    #voc=km.cluster_centers_
    return voc

#voc=create_voc(descriptors,k=10)

def trsf_BoF(des_list, voc):
    # Calculate the histogram of features
    im_features = np.zeros((len(des_list),len(voc)), "float32")
    for i in range(len(des_list)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1
    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)
    return im_features

#im_features=trsf_BoF(des_list,voc)
