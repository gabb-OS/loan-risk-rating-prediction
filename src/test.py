MY_UNIQUE_ID = "AprileTassi"

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
import pickle
import pandas as pd
import numpy as np
import os
import torch
import sys
from support_modules.utils import *
from support_modules.preprocessing import *
# ========== CONFIGURA SKLEARN PER OUTPUT PANDAS ==========
from sklearn import set_config
set_config(transform_output="pandas")  # ← AGGIUNGI QUESTA RIGA
# =========================================================

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

    # Da dizionario in input a dataframe
    data = pd.DataFrame.from_dict(dataset)

    # Drop duplicates
    df_undup = remove_duplicates(data)

    # Split
    X = df_undup.drop(columns=["grade"])
    y = df_undup["grade"]

    # Encoding Label 
    le = LabelEncoder()
    y = le.fit_transform(y)


    dataset_processed = {}
    preprocessor = None
    
    if clfName == "rf":
        preprocessor = pickle.load(open("rf_preprocessor.save", 'rb')) 
    elif clfName == "svm":
        preprocessor = pickle.load(open("svm_preprocessor.save", 'rb'))
    elif clfName == "knn":
        preprocessor = pickle.load(open("knn_preprocessor.save", 'rb'))
    elif clfName == "ff":
        preprocessor = pickle.load(open("ff_preprocessor.save", 'rb'))
    elif clfName == "tb":
        preprocessor = pickle.load(open("tb_preprocessor.save", 'rb'))
    elif clfName == "tt":
        #preprocessor = pickle.load(open("tt_preprocessor.save", 'rb'))
        print("Model not trained")

    if preprocessor is not None:
        try:
            X_transformed = preprocessor.transform(X)
            dataset_processed['data'] = X_transformed
            dataset_processed['grade'] = y
        except Exception as e:
            print(f"ERRORE durante transform: {e}")
            print(f"Tipo di X passato al preprocessor: {type(X)}")
            raise
    else:
        dataset_processed['data'] = X.values
        dataset_processed['grade'] = y

    return dataset_processed


# Input: Classifier name ("svc": Support Vector Classifier, ecc)
# Output: Classifier object
def load(clfName):
    device = getDevice()
    clf = None
    
    if clfName == "rf":
        clf = pickle.load(open("rf.save", 'rb'))    
    elif clfName == "svm":
        clf = pickle.load(open("svm.save", 'rb'))
    elif clfName == "knn":
        clf = pickle.load(open("knn.save", 'rb'))
    elif clfName == "ff":
        # TODO da FARE
        clf = FeedForward_NN(input_size, num_classes, hidden_size, dropout_rate, depth=1).to(device)
        checkpoint = torch.load('ff.pth', map_location=device)
        clf.load_state_dict(checkpoint)
    elif clfName == "tb":
        clf = TabNetClassifier()
        clf.load_model('tabnet_best_model.zip')
        # TODO: implementa TabNet
        print("TabNet not implementata")
        clf = None

    return clf


# Input: PreProcessed Dataset dictionary, Classifier Name, Classifier Object 
# Output: Performance dictionary
def predict(dataset, clf):
    X = dataset['data']
    y = dataset['grade']
    

    # TODO PREDICT IN FF         clf.eval()
    
    ypred = clf.predict(X)

    acc = accuracy_score(y, ypred)
    bacc = balanced_accuracy_score(y, ypred)
    f1 = f1_score(y, ypred, average="weighted")
    
    perf = {"acc": acc, "bacc": bacc, "f1": f1}
    
    return perf
    
