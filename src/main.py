import pandas as pd
from sklearn.model_selection import train_test_split
import test

def run_simulation():
    SEED = 42
    print(f"--- Avvio Simulazione per Team: {test.getName()} ---\n")

    # 1. Caricamento Dataset
    try:
        df = pd.read_csv('../data/train.csv', low_memory=False)
    except FileNotFoundError:
        print("Errore: File data/train.csv non trovato.")
        return

    X = df.drop(columns=['grade'])
    y = df['grade']

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    # dizionario
    X_val['grade'] = y_val
    dataset_test = X_val.to_dict()

    
    # Elenco algoritmi da testare
    #algorithms = ["knn", "svm", "rf", "ff", "tb"]
    algorithms = ["tb"]
    
    all_results = {}

    for clfName in algorithms:
        print(f"Esecuzione test per: {clfName}...")
        
        try:
            dataset_processed = test.preprocess(dataset_test, clfName)

            clf_object = test.load(clfName)
            
            if clf_object is None:
                print(f"  [!] Modello {clfName} non ancora implementato o file mancante.\n")
                continue

            performance = test.predict(dataset_processed, clf_object)
            
            all_results[clfName] = performance
            print(f"  [+] Risultati {clfName}: {performance}\n")

        except Exception as e:
            print(f"  [X] Errore durante il test di {clfName}: {e}\n")

    print("--- Riepilogo Finale ---")
    print(pd.DataFrame(all_results).T)

if __name__ == "__main__":
    run_simulation()