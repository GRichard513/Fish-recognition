from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.cluster.vq import *
import xgboost as xgb

from utils import *
from SIFT_extraction import *
from BoF import *


def main(nb_split=10, nb_features=500,output_name='submission', CV=False):
    #Load paths
    fnames, file_paths, cat_int, int_cat, training_names, train_path, test_names=load_paths()
    image_paths, image_classes=load_images(train_path,training_names)

    #Shuffle and split images
    zz=np.array([[image_classes[i], image_paths[i]] for i in range(len(image_classes))])
    np.random.shuffle(zz)
    image_paths=zz.T[1]
    image_classes=zz.T[0]
    list_img=np.array_split(image_paths,nb_split)

    #Extract SIFT and build vocabulary
    voc=[]
    des_list=[]
    it=1
    for sublist in list_img:
        print('Extraction: %i / %i'%(it,len(list_img)))
        des=SIFT_extractor(sublist)
        des_list.append(des)
        if it==1:
            descriptors, _=transform_descriptors(des,sublist)
            print('KMeans for features')
            voc.append(create_voc(descriptors,k=nb_features))
        it=it+1
        print('-----------------')
    #Aggregation
    voc_tot=np.vstack(voc)
    des_tot=[]
    for i in range(len(des_list)):
        des_tot=des_tot+des_list[i]

    #Build histogram
    print('Histogram building')
    im_features=trsf_BoF(des_tot,voc_tot)
    print('-----------------')

    #Fit SVC
    print('Training of classifier')
    clf = xgb.XGBClassifier(n_estimators=500,max_depth=5)
    if CV:
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        clf=GridSearchCV(clf,tuned_parameters, cv=5)
    clf.fit(im_features, np.array(image_classes))
    print('-----------------')

    #Test application
    print('Test')
    test_paths=['./test_stg1/'+t for t in test_names]
    print('---------------')
    print('SIFT extraction')
    print('---------------')
    des_test=SIFT_extractor(test_paths)
    test_features= trsf_BoF(des_test,voc_tot)
    output_proba=clf.predict_proba(test_features)
    result_write=[output_proba[i].tolist()+[test_names[i]] for i in range(len(test_names))]
    result_to_csv(output_name, result_write)
    return #im_features, np.array(image_classes), voc_tot

#im_features, image_classes, voc=
main(10,200,'submission', False)
