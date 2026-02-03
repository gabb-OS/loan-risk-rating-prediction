MY_UNIQUE_ID = "DaiCheOggiLaChiudiamo"

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import pickle
import pandas as pd
import numpy as np
import os

from preprocessing import *
'''
Il campo clfName è una stringa con i seguenti valori ammissibili:
- 'knn' → identifica classificatore K-Nearest Neighbour
- 'rf' → identifica classificatore Random Forest
- 'svm' → identifica classificatore Support Vector Machine
- 'ff' → identificare classificatore con reti neurali, architettura Feed Forward
- 'tb' → identificare classificatore con reti neurali, architettura TabNet
- 'tf' → identificare classificatore con reti neurali, architettura TabTransformer
'''

# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID


# Input: Dataset dictionary and classifier name
# Output: PreProcessed Dataset dictionary
def preprocess(dataset, clfName):

    # Drop duplicates
    remove_duplicates(dataset)

    X = dataset.drop(columns=["grade"])
    y = dataset["grade"]

    # Drop columns 
    drop_leakage_and_non_significant_cols(X)
    drop_high_nan_cols(X)
    drop_joint_and_secondary_cols(X)
    drop_higly_correlated_numeric_features(X)

    # Feature extraction
    feature_extraction(X)

    # Rounding features
    round_features_to_int(X)

    # NaN management
    nan_management_general_fill(X)
    nan_management_imputation_fill(X)

    # Outliers management
    apply_capping(X)
    apply_log_transform_on_skewed_cols(X)

    #### TODO: ENCODING
    #here encoding

    dataset_processed = {}
    scaler = None
    
    if clfName == "rf":
        scaler = pickle.load(open("rf_scaler.save", 'rb'))    
    elif clfName == "svm":
        scaler = pickle.load(open("svm_scaler.save", 'rb'))
    elif clfName == "knn":
        scaler = pickle.load(open("knn_scaler.save", 'rb'))
    elif clfName == "ff":
        scaler = pickle.load(open("ff_scaler.save", 'rb'))
    elif clfName == "tb":
        scaler = pickle.load(open("tb_scaler.save", 'rb'))
    elif clfName == "tt":
        scaler = pickle.load(open("tt_scaler.save", 'rb'))
        

    if scaler is not None:
        dataset_processed = scaler.transform(X)
        dataset_processed['grade'] = y
    else:
        # ????? TODO CHECK
        dataset_processed = X
        dataset_processed["grade"] = y



    
    return dataset_processed


# Input: Classifier name ("lr": Logistic Regression, "svc": Support Vector Classifier)
# Output: Classifier object
def load(clfName):
    clf = None
    
    if clfName == "rf":
        clf = pickle.load(open("rf.save", 'rb'))    
    elif clfName == "svm":
        clf = pickle.load(open("svm.save", 'rb'))
    elif clfName == "knn":
        clf = pickle.load(open("knn.save", 'rb'))
    elif clfName == "ff":
        clf = pickle.load(open("ff.save", 'rb'))
    elif clfName == "tb":
        clf = pickle.load(open("tb.save", 'rb'))
    elif clfName == "tt":
        clf = pickle.load(open("tt.save", 'rb'))

    return clf


# Input: PreProcessed Dataset dictionary, Classifier Name, Classifier Object 
# Output: Performance dictionary
def predict(dataset, clf):
    X = dataset['data']
    y = dataset['target']
    
    ypred = clf.predict(X)

    acc = accuracy_score(y, ypred)
    bacc = balanced_accuracy_score(y, ypred)
    f1 = f1_score(y, ypred, average="weighted")
    
    perf = {"acc": acc, "bacc": bacc, "f1": f1}
    
    return perf
    
