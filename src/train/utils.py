import pandas as pd

import os
import pickle

def drop_high_nan_columns(df, threshold=0.95):
    min_valid_values = (1-threshold)*len(df)

    cols_to_drop = df.columns[df.notna().sum() < min_valid_values].tolist()

    if cols_to_drop:
        print(f"Colonne rimosse (> {threshold*100}% NaN):")
        print(cols_to_drop)
    else:
        print("Nessuna colonna rimossa.")

    return df.dropna(thresh=min_valid_values, axis=1)


def print_nan(df, types=None):
    """
    Esplora i NaN e i valori univoci (mostrando i valori effettivi) per ogni feature.
    """
    # Selezione colonne per tipo
    if types:
        selected_cols = df.select_dtypes(include=types).columns
    else:
        selected_cols = df.columns

    working_df = df[selected_cols]

    # Liste per raccogliere i dati
    data = []

    for col in selected_cols:
        # Calcolo metriche per la colonna
        nan_count = working_df[col].isna().sum()
        nan_perc = (nan_count / len(df)) * 100
        dtype = working_df[col].dtype

        # Otteniamo i valori unici (escludendo i NaN per chiarezza)
        uniques = working_df[col].dropna().unique()
        n_uniques = len(uniques)

        # Formattazione della stringa dei valori unici
        if n_uniques <= 10:
            uniques_str = str(list(uniques))
        else:
            # Se sono troppi, mostriamo un'anteprima
            uniques_str = f"{list(uniques[:5])}... (+{n_uniques-5} more)"

        data.append({
            'Feature': col,
            'Type': dtype,
            'Nan': nan_count,
            'Percentuale NaN (%)': f"{nan_perc:.2f}%",
            'Uniques Count': n_uniques,
            'Unique Values': uniques_str
        })

    # Creazione e ordinamento della tabella
    nan_table = pd.DataFrame(data).sort_values(by='Uniques Count')

    # Stampa con formattazione migliorata
    print(nan_table.to_string(index=False))
    return


def calculate_outlier_percentage(df):
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Create a list to store results
    results = []

    for col in numeric_cols:
        # 1. Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 2. Count Outliers
        # An outlier is anything strictly less than lower OR strictly greater than upper
        n_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]

        # 3. Calculate Percentage
        total_rows = df.shape[0]
        percentage = (n_outliers / total_rows) * 100

        results.append({
            "Column": col,
            "Lower Cutoff": round(lower_bound, 2),
            "Upper Cutoff": round(upper_bound, 2),
            "Outliers Count": n_outliers,
            "Outliers %": round(percentage, 2)
        })

    # Convert to DataFrame for a nice table display
    results_df = pd.DataFrame(results)

    # Sort by percentage descending to see the "worst" columns first
    results_df = results_df.sort_values(by="Outliers %", ascending=False)

    return results_df


def evaluate_model(model_input, X_test, y_test, model_name=None):
    """
    Valuta un modello, stampa i parametri migliori e mostra la matrice di confusione.
    Accetta: oggetti GridSearchCV, Pipeline/Stimatori o percorsi file (.save).
    """
    params = None

    # 1. Gestione dell'input (File, GridSearch o Modello diretto)
    if isinstance(model_input, str):
        # Caricamento da FILE
        if not os.path.exists(model_input):
            print(f"Errore: Il file '{model_input}' non esiste.")
            return None, None
        with open(model_input, "rb") as f:
            model = pickle.load(f)
        if model_name is None: model_name = f"File: {model_input}"
        # Per i file caricati, prendiamo i parametri attuali dell'oggetto
        params = model.get_params() if hasattr(model, 'get_params') else "Non disponibili"

    elif isinstance(model_input, GridSearchCV):
        # Input è l'oggetto GRID SEARCH
        model = model_input.best_estimator_
        params = model_input.best_params_ # Qui abbiamo i parametri specifici scelti dalla ricerca
        if model_name is None: model_name = "Best GridSearch Model"

    else:
        # Input è il MODELLO/PIPELINE in memoria
        model = model_input
        params = model.get_params() if hasattr(model, 'get_params') else "Non disponibili"
        if model_name is None: model_name = "Model in memory"

    # 2. Predizione
    y_pred = model.predict(X_test)

    # 3. Probabilità/Score per metriche future
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)

    # --- OUTPUT ---
    print(f"\n{'='*30}")
    print(f" EVALUATION: {model_name}")
    print(f"{'='*30}")

    # STAMPA PARAMETRI
    print("\n>>> BEST/CURRENT PARAMETERS:")
    if isinstance(params, dict):
        # Stampiamo solo i parametri salienti (quelli che iniziano con 'clf__')
        # per non intasare l'output con i parametri dello scaler o di SMOTE
        relevant_params = {k: v for k, v in params.items() if 'clf__' in k or not '__' in k}
        for k, v in relevant_params.items():
            print(f"  - {k}: {v}")
    else:
        print(f"  {params}")

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")

    # --- MATRICE DI CONFUSIONE ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    classes = sorted(list(set(y_test)))
    plt.xticks(ticks=[i + 0.5 for i in range(len(classes))], labels=classes)
    plt.yticks(ticks=[i + 0.5 for i in range(len(classes))], labels=classes, rotation=0)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

    return


def evaluate_knn(model_input, X_test, y_test, model_name=None):
    """
    Valuta un modello KNN, stampa i parametri specifici (K, pesi, metrica) 
    e mostra la matrice di confusione.
    
    Args:
        model_input: Può essere un percorso file (.pkl), un oggetto GridSearchCV o un Estimator/Pipeline.
        X_test: Feature di test (IMPORTANTE: Devono essere scalate se il modello non è una Pipeline che include lo scaling).
        y_test: Target reali.
        model_name: Nome opzionale per il grafico.
    """
    params = None
    model = None
    k_value = "?"

    # 1. Gestione dell'input (File, GridSearch o Modello diretto)
    if isinstance(model_input, str):
        # Caricamento da FILE
        if not os.path.exists(model_input):
            print(f"Errore: Il file '{model_input}' non esiste.")
            return
        with open(model_input, "rb") as f:
            model = pickle.load(f)
        if model_name is None: model_name = f"File: {os.path.basename(model_input)}"
        params = model.get_params()

    elif isinstance(model_input, GridSearchCV):
        # Input è l'oggetto GRID SEARCH
        model = model_input.best_estimator_
        params = model_input.best_params_
        if model_name is None: model_name = "Best KNN (GridSearch)"

    else:
        # Input è il MODELLO/PIPELINE in memoria
        model = model_input
        params = model.get_params()
        if model_name is None: model_name = "KNN Model"

    # 2. Predizione
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Errore durante la predizione: {e}")
        return

    # 3. Estrazione parametri rilevanti per KNN
    # Cerchiamo di isolare i parametri chiave del KNN anche se è dentro una Pipeline
    knn_params = {}
    if isinstance(params, dict):
        # Lista di keyword tipiche del KNN
        target_keys = ['n_neighbors', 'weights', 'metric', 'p', 'leaf_size', 'algorithm']
        
        for k, v in params.items():
            # Controlla se la chiave corrisponde o finisce con una delle target_keys (es. 'knn__n_neighbors')
            if any(key in k.split('__')[-1] for key in target_keys):
                clean_key = k.split('__')[-1] # Rimuove il prefisso della pipeline se c'è
                knn_params[clean_key] = v
                if clean_key == 'n_neighbors':
                    k_value = v
    
    # --- OUTPUT ---
    print(f"\n{'='*40}")
    print(f" KNN EVALUATION: {model_name}")
    print(f"{'='*40}")

    # STAMPA PARAMETRI
    print("\n>>> CONFIGURAZIONE KNN:")
    if knn_params:
        for k, v in knn_params.items():
            print(f"  - {k}: {v}")
    else:
        print("  Parametri specifici non trovati (o modello custom).")
        print(f"  Dump parametri completi: {params}")

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    # --- MATRICE DI CONFUSIONE ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    classes = sorted(list(set(y_test)))
    # Centratura labels
    plt.xticks(ticks=[i + 0.5 for i in range(len(classes))], labels=classes)
    plt.yticks(ticks=[i + 0.5 for i in range(len(classes))], labels=classes, rotation=0)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f"Confusion Matrix: {model_name}\n(K={k_value}, Acc={acc:.2f})")
    plt.show()

    return model