MY_UNIQUE_ID = "AprileTassi"
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
import pickle
import pandas as pd
import numpy as np
import torch
from support_modules.utils import *
from support_modules.preprocessing import *
from support_modules.ff_utils import *

from sklearn import set_config
set_config(transform_output="pandas")

'''
Il campo clfName è una stringa con i seguenti valori ammissibili:
- 'knn' → identifica classificatore K-Nearest Neighbour
- 'rf' → identifica classificatore Random Forest
- 'svm' → identifica classificatore Support Vector Machine
- 'ff' → identificare classificatore con reti neurali, architettura Feed Forward
- 'tb' → identificare classificatore con reti neurali, architettura TabNet
- 'tf' → identificare classificatore con reti neurali, architettura TabTransformer
'''

def getName():
    return MY_UNIQUE_ID


def preprocess(dataset, clfName):

    data = pd.DataFrame.from_dict(dataset)

    df_undup = remove_duplicates(data)

    X = df_undup.drop(columns=["grade"])
    y = df_undup["grade"]

    le = LabelEncoder()
    # hardcoding delle label per usare lo stesso mapping del train
    le.classes_ = np.array(["A", "B", "C", "D", "E", "F", "G"])
    y = le.transform(y)


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
    elif clfName == "tf":
        print("Modello non implementato")

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
        metadata = pickle.load(open("ff_metadata.save", 'rb'))
        input_size = metadata['input_size']
        num_classes = metadata['num_classes']
        hidden_size = metadata.get('hidden_size')
        dropout_rate = metadata.get('dropout_rate')
        depth = metadata.get('depth', 1)

        clf = FeedForward_NN(input_size, num_classes, hidden_size, dropout_rate, depth).to(device)
        checkpoint = torch.load('ff.pth', map_location=device)
        clf.load_state_dict(checkpoint)
    elif clfName == "tb":
        clf = TabNetClassifier()
        clf.load_model('tb.zip')
    elif clfName == "tf":
        print("TabTransformer non implementata")
        clf = None

    return clf


def predict(dataset, clf):
    X = dataset['data']
    y = dataset['grade']

    if isinstance(clf, FeedForward_NN):
        device = getDevice()
        clf.eval()

        if isinstance(X, pd.DataFrame):
            X_tensor = torch.FloatTensor(X.values).to(device)
        else:
            X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            outputs = clf(X_tensor)
            _, ypred = torch.max(outputs, 1)
            ypred = ypred.cpu().numpy()
    
    elif isinstance(clf, TabNetClassifier):
        ypred = clf.predict(X)
    
    else:
        # classificatori sklearn (KNN, RF, SVM)
        ypred = clf.predict(X)

    acc = accuracy_score(y, ypred)
    bacc = balanced_accuracy_score(y, ypred)
    f1 = f1_score(y, ypred, average="weighted")
    
    perf = {"acc": acc, "bacc": bacc, "f1": f1}
    
    return perf