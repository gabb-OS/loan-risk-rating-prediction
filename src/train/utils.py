import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import seaborn as sns
import os
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
    PrecisionRecallDisplay
)

def drop_high_nan_columns(df, threshold):
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


def evaluate_model(X_val_raw, y_val_true, prefix):
    """
    prefix: "knn" o "svc" a seconda dei file salvati
    """
    # 1. Caricamento dinamico dei file
    try:
        with open(f"{prefix}_scaler.save", "rb") as f:
            scaler = pickle.load(f)
        with open(f"{prefix}_pca.save", "rb") as f:
            pca = pickle.load(f)
        with open(f"{prefix}.save", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Errore: File con prefisso '{prefix}' non trovati. {e}")
        return

    print(f"========================================")
    print(f"REPORT VALUTAZIONE: {prefix.upper()}")
    print(f"========================================")
    
    # 2. Stampa parametri specifici per tipo di modello
    print(f"Scaler usato: {type(scaler).__name__}")
    print(f"Componenti PCA: {pca.n_components_} ({pca.explained_variance_ratio_.sum():.2%} varianza)")
    
    if prefix == "knn":
        print(f"Parametri KNN: n_neighbors={model.n_neighbors}, weights='{model.weights}', metric='{model.metric}'")
    else:
        print(f"Parametri SVC: C={model.C}, kernel='{getattr(model, 'kernel', 'linear')}', class_weight='{model.class_weight}'")
    
    print(f"----------------------------------------\n")

    # 3. Trasformazione e Predizione (struttura identica)
    X_scaled = scaler.transform(X_val_raw)
    X_pca = pca.transform(X_scaled)
    y_pred = model.predict(X_pca)

    # 4. Metriche
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_val_true, y_pred))
    
    print(f"Accuracy Score:          {accuracy_score(y_val_true, y_pred):.4f}")
    print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_val_true, y_pred):.4f}")

    # 5. Matrice di Confusione
    cm = confusion_matrix(y_val_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Greens' if prefix == "knn" else 'Blues', ax=ax, values_format='d')
    plt.title(f'Confusion Matrix - {prefix.upper()}')
    plt.show()




def apply_capping(df, lower_quantile, upper_quantile):
    """
    Applica la tecnica del Capping (Winsorization) alle colonne numeriche di un DataFrame.
    I valori sotto il lower_quantile vengono sostituiti con il valore del lower_quantile.
    I valori sopra l'upper_quantile vengono sostituiti con il valore dell'upper_quantile.

    Parametri:
    - df: DataFrame in input
    - lower_quantile: soglia inferiore (default 0.01 -> 1%)
    - upper_quantile: soglia superiore (default 0.99 -> 99%)

    Return:
    - DataFrame con i valori cappati.
    """
    # Creiamo una copia per non modificare l'originale in-place
    df_capped = df.copy()

    # Selezioniamo solo le colonne numeriche
    numerical_cols = df_capped.select_dtypes(include=['float', 'int']).columns

    for col in numerical_cols:
        # Calcolo dei limiti
        lower_limit = df_capped[col].quantile(lower_quantile)
        upper_limit = df_capped[col].quantile(upper_quantile)

        # Applicazione del capping (clipping)
        # I valori < lower_limit diventano lower_limit
        # I valori > upper_limit diventano upper_limit
        df_capped[col] = df_capped[col].clip(lower=lower_limit, upper=upper_limit)

        # Opzionale: Stampa per vedere l'effetto (puoi commentarlo)
        print(f"Colonna '{col}': cappata tra {lower_limit:.2f} e {upper_limit:.2f}")

    return df_capped







def identify_distributions(df, threshold_skew, threshold_peaks_prominence):
    # 1 / 0.05
    """
    Identifica se le colonne numeriche sono Skewed, Multimodali, Uniformi o Normali.
    Restituisce un DataFrame con i risultati dell'analisi.
    """
    results = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        data = df[col].dropna()
        if len(data) < 50: continue # Salta colonne con troppi pochi dati

        # 1. Check Skewness (Asimmetria)
        skew_val = data.skew()
        is_skewed = abs(skew_val) > threshold_skew

        # 2. Check Normality (Test di D'Agostino's K-squared)
        try:
            k2, p_norm = stats.normaltest(data)
            # p_value > 0.01 e skewness bassa indicano normalità
            is_normal = (p_norm > 0.01) and (abs(skew_val) < 0.5)
        except:
            is_normal = False

        # 3. Check Multimodality (Picchi nell'istogramma)
        # Calcola la densità approssimata tramite istogramma
        counts, bin_edges = np.histogram(data, bins='auto', density=True)
        # Trova picchi che siano rilevanti (almeno il 5% della densità massima)
        peaks, _ = find_peaks(counts, prominence=np.max(counts) * threshold_peaks_prominence)
        num_peaks = len(peaks)
        is_multimodal = num_peaks > 1

        # 4. Check Uniformity (Varianza delle frequenze)
        # Se la deviazione standard delle frequenze nei bin è molto bassa, è uniforme
        counts_uni, _ = np.histogram(data, bins=20)
        cv = np.std(counts_uni) / np.mean(counts_uni)
        is_uniform = cv < 0.2 # Soglia euristica: deviazione < 20% della media

        # Logica di Classificazione (Priorità)
        classification = "Unknown"
        if is_uniform:
            classification = "Uniform"
        elif is_multimodal and not is_skewed:
             # Spesso le code lunghe creano falsi picchi, quindi diamo priorità allo skew
             classification = "Multimodal"
        elif is_skewed:
            classification = "Positively Skewed" if skew_val > 0 else "Negatively Skewed"
        elif is_normal:
            classification = "Normal"
        else:
             classification = "Symmetric (Non-Normal)"

        results.append({
            'Feature': col,
            'Skewness': round(skew_val, 2),
            'Num_Peaks': num_peaks,
            'Classification': classification
        })

    return pd.DataFrame(results)





def get_distribution_type(data, threshold_skew, threshold_peaks):
    # 1/0.05z
    """
    Funzione helper per identificare il tipo di distribuzione di una singola Series.
    """
    # Rimuoviamo NaN per l'analisi
    clean_data = data.dropna()
    if len(clean_data) < 50: return 'Other'

    # 1. Calcolo Skewness
    skew_val = clean_data.skew()

    # 2. Test Normalità (D'Agostino's K-squared test)
    try:
        k2, p_norm = stats.normaltest(clean_data)
        # Se p > 0.01 e skewness è bassa, consideriamo "Normale"
        is_normal = (p_norm > 0.01) and (abs(skew_val) < 0.5)
    except:
        is_normal = False

    if is_normal:
        return 'Normal'

    # 3. Controllo Skewness (Priorità alta)
    if abs(skew_val) > threshold_skew:
        return 'Skewed'

    # 4. Controllo Multimodalità (Picchi)
    counts, bin_edges = np.histogram(clean_data, bins='auto', density=True)
    peaks, _ = find_peaks(counts, prominence=np.max(counts) * threshold_peaks)

    if len(peaks) > 1:
        return 'Multimodal'

    return 'Other' # Simmetrica ma non normale, o altro

def auto_transform_features(df):
    """
    Analizza ogni colonna numerica e applica la trasformazione:
    - Skewed -> Log Transformation
    - Normal -> Z-Score Standardization
    - Multimodal -> Binning (5 Quantiles)
    """
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()

    # Report per tenere traccia delle modifiche
    report_list = []

    for col in num_cols:
        # Identifica la distribuzione
        dist_type = get_distribution_type(df_clean[col])
        action = "Nessuna azione"

        # --- LOGICA DI TRASFORMAZIONE ---

        if dist_type == 'Skewed':
            # LOG TRANSFORMATION
            # Gestione valori negativi/zero: trasliamo se necessario
            min_val = df_clean[col].min()
            if min_val <= 0:
                offset = abs(min_val) + 1
                df_clean[col] = np.log(df_clean[col] + offset)
                action = f"Log Transform (Shifted +{offset})"
            else:
                df_clean[col] = np.log(df_clean[col])
                action = "Log Transform"

        elif dist_type == 'Normal':
            # Z-SCORE STANDARDIZATION
            # Reshape necessario per StandardScaler (n_samples, 1)
            values = df_clean[col].values.reshape(-1, 1)
            df_clean[col] = scaler.fit_transform(values)
            action = "Z-Score Standardization"

        elif dist_type == 'Multimodal':
            # BINNING
            # Usiamo qcut per dividere in 5 quantili (0,1,2,3,4)
            # duplicates='drop' gestisce casi in cui molti valori sono identici
            try:
                df_clean[col] = pd.qcut(df_clean[col], q=5, labels=False, duplicates='drop')
                action = "Binning (5 Quantiles)"
            except Exception as e:
                action = f"Binning Fallito: {e}"

        # Salviamo il report
        report_list.append({
            'Feature': col,
            'Tipo Rilevato': dist_type,
            'Trasformazione': action
        })

    report_df = pd.DataFrame(report_list)
    return df_clean, report_df

