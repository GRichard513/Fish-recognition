import numpy as np
import pandas as pd
import cv2
import csv
import os
from os import listdir
from os.path import isfile, join
import glob
#import matplotlib.pyplot as plt
from subprocess import check_output

def load_paths():
    #print('----Check output----\n', check_output(["ls", "../train/"]).decode("utf8"))

    # Any results you write to the current directory are saved as output.
    dir_names = [d for d in listdir("../train/") if not isfile(join("../train/", d))]
    #print('\n ----Subdirectories----\n', dir_names)

    #file_paths contains the file names for each category
    #key has format: (Category,CatNumber,Path)
    file_paths = {}
    class_num = 0
    cat_int = {}
    int_cat = {}
    prediction_output_list=[]
    test_names=[f for f in listdir("../test_stg1/") if isfile(join("../test_stg1", f))]
    train_path = '../train'
    training_names = os.listdir(train_path)
    for d in dir_names:
        fnames = [f for f in listdir("../train/"+d+"/") if isfile(join("../train/"+d+"/", f))]
        file_paths[(d, class_num, "../train/"+d+"/")] = fnames
        cat_int[d]=class_num
        int_cat[class_num]=d
        class_num += 1
    return fnames, file_paths, cat_int, int_cat, training_names, train_path, test_names


#fnames, file_paths, cat_int, int_cat, training_names, train_path, test_names=load_paths()

def load_images(path,names,k=-1):
    if k==-1:
        k=1000000
    image_paths = []
    image_classes = []
    class_id = 0
    for name in names:
        dir = os.path.join(path, name)
        class_path = [dir+'/'+f for f in listdir(dir) if isfile(join(dir, f))]
        class_path=class_path[:k]
        image_paths+=class_path
        image_classes+=[class_id]*len(class_path)
        class_id+=1
    return image_paths, image_classes

#image_paths, image_classes=load_images(train_path,training_names,k=-1)

def result_to_csv(output_name, result):
    try:
        with open(output_name+".csv", "w") as f:
            writer = csv.writer(f)
            f.flush()
            headers = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT','image']
            #headers= "image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT"
            writer.writerow(headers)
            writer.writerows(result)
    finally:
        f.close()
