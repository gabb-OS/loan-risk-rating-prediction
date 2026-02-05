import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Importiamo le funzioni dal tuo file test.py
import test

def run_simulation():
    SEED = 42
    print(f"--- Avvio Simulazione per Team: {test.getName()} ---\n")

    # 1. Caricamento Dataset
    try:
        # Assicurati che il percorso sia corretto rispetto alla tua struttura cartelle
        df = pd.read_csv('../data/train.csv', low_memory=False)
    except FileNotFoundError:
        print("Errore: File data/train.csv non trovato.")
        return

    # 2. Definizione X e y (grade è il target per il rischio finanziario)
    X = df.drop(columns=['grade'])
    y = df['grade']

    # 3. Splitting richiesto: usa il validation come test set per la prova
    # test_size=0.25, stratify=y, random_state=42
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    # Prepariamo il dizionario simulando i dati di TEST
    X_val['grade'] = y_val
    dataset_test = X_val.to_dict()
    
    """  dataset_test = {
        'data': X_val,
        'target': y_val
    } """
    
    # Elenco algoritmi da testare
    #algorithms = ["knn", "svm", "rf", "ff", "tb"]
    algorithms = ["tb"]
    
    all_results = {}

    for clfName in algorithms:
        print(f"Esecuzione test per: {clfName}...")
        
        try:
            # FASE 1: Preprocessing
            # Carica i parametri (scaler, imputer) salvati in training e trasforma i dati di test
            dataset_processed = test.preprocess(dataset_test, clfName)
            
            # FASE 2: Load
            # Istanzia il classificatore caricando il file .pkl o .model corrispondente
            clf_object = test.load(clfName)
            
            if clf_object is None:
                print(f"  [!] Modello {clfName} non ancora implementato o file mancante.\n")
                continue

            # FASE 3: Predict
            # Esegue la predizione e calcola le performance (Acc, Balanced Acc, F1)
            performance = test.predict(dataset_processed, clf_object)
            
            all_results[clfName] = performance
            print(f"  [+] Risultati {clfName}: {performance}\n")

        except Exception as e:
            print(f"  [X] Errore durante il test di {clfName}: {e}\n")

    # Visualizzazione riepilogativa
    print("--- Riepilogo Finale ---")
    print(pd.DataFrame(all_results).T)

if __name__ == "__main__":
    run_simulation()