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
)


def drop_high_nan_columns(df, threshold):
    # percentuale di NaN per colonna
    nan_ratio = df.isna().mean() * 100

    # colonne da rimuovere
    cols_to_drop = nan_ratio[nan_ratio > threshold * 100].index.tolist()

    if cols_to_drop:
        print(f"Colonne rimosse (> {threshold*100}% NaN):")
        print(cols_to_drop)
    else:
        print("Nessuna colonna rimossa.")

    return df.drop(columns=cols_to_drop)



def print_high_nan_columns(df, threshold):
    min_valid_values = (1-threshold)*len(df)

    cols_to_drop = df.columns[df.notna().sum() < min_valid_values].tolist()

    if cols_to_drop:
        print(f"Colonne con > {threshold*100}% NaN:")
        print(cols_to_drop)
    else:
        print("Nessuna colonna rimossa.")
    return


def print_nan(df, types=None, sort_by='nan', ascending=False, show_values=True):
    """
    Esplora i NaN (e opzionalmente i valori univoci) per ogni feature.
    Mostra solo le colonne che contengono almeno un valore NaN.

    Parametri
    ---------
    types : list o None
        Tipi da includere (es. ['number', 'object']). None = tutte le colonne.
    sort_by : str
        Caratteristica su cui ordinare: 'feature', 'type', 'nan', 'perc', 'uniques'.
    ascending : bool
        Ordine crescente (True) o decrescente (False).
    show_values : bool
        Se True mostra la colonna con i valori univoci, altrimenti la nasconde.
    """
    sort_map = {
        'feature':  'Feature',
        'type':     '_type_str',
        'nan':      'Nan',
        'perc':     '_nan_perc',
        'uniques':  'Uniques Count',
    }
    if sort_by not in sort_map:
        raise ValueError(f"sort_by deve essere uno tra: {list(sort_map)}")

    selected_cols = df.select_dtypes(include=types).columns if types else df.columns
    n_total_cols = len(selected_cols)

    data = []
    for col in selected_cols:
        nan_count = df[col].isna().sum()
        if nan_count == 0:
            continue

        nan_perc = nan_count / len(df) * 100
        uniques = df[col].dropna().unique()
        n_uniques = len(uniques)

        try:
            uniques = sorted(uniques)
        except TypeError:
            uniques = list(uniques)

        if n_uniques <= 10:
            uniques_str = ", ".join(map(str, uniques))
        else:
            preview = ", ".join(map(str, uniques[:5]))
            uniques_str = f"{preview}, … (+{n_uniques - 5})"

        data.append({
            'Feature': col,
            'Type': str(df[col].dtype),
            'Nan': nan_count,
            'Percentuale NaN (%)': f"{nan_perc:.2f}%",
            'Uniques Count': n_uniques,
            'Unique Values': uniques_str,
            '_type_str': str(df[col].dtype),
            '_nan_perc': nan_perc,
        })

    # --- Riga di riepilogo in cima ---
    n_cols_with_nan = len(data)
    cols_perc = n_cols_with_nan / n_total_cols * 100 if n_total_cols else 0
    print(f"Colonne con almeno un NaN: {n_cols_with_nan}/{n_total_cols} "
          f"({cols_perc:.2f}%)")
    print("-" * 60)

    if not data:
        print("Nessuna colonna contiene valori NaN.")
        return

    nan_table = pd.DataFrame(data).sort_values(by=sort_map[sort_by],
                                               ascending=ascending)

    cols_to_show = ['Feature', 'Type', 'Nan', 'Percentuale NaN (%)', 'Uniques Count']
    if show_values:
        cols_to_show.append('Unique Values')

    nan_table = nan_table[cols_to_show]

    with pd.option_context('display.max_colwidth', 60,
                           'display.colheader_justify', 'left'):
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


def evaluate_model(X_val_raw, y_val_true, clfName, model_dir="models", preproc_dir="preprocessing"):
    """
    Valuta il modello applicando ESATTAMENTE la stessa pipeline di preprocessing
    usata in training (stessa logica di preprocess()).

    Parametri
    ---------
    X_val_raw : DataFrame delle feature di validazione (grezze, non trasformate)
    y_val_true : target di validazione (già label-encoded, coerente col training)
    clfName : 'rf', 'svm', 'knn', 'ff', 'tb', 'tt'
    model_dir : cartella dei file .save del modello
    preproc_dir : cartella dei file _preprocessor.save
    """

    # 1. Caricamento del preprocessor
    preprocessor = None
    preproc_path = os.path.join(preproc_dir, f"{clfName}_preprocessor.save")

    if clfName in ("rf", "rf_new", "svm", "knn", "ff", "tb"):
        if os.path.exists(preproc_path):
            preprocessor = pickle.load(open(preproc_path, 'rb'))
        else:
            print(f"Attenzione: preprocessor non trovato per '{clfName}' in '{preproc_path}'")
    elif clfName == "tt":
        print("Model not trained")
        return
    else:
        print(f"clfName '{clfName}' non riconosciuto")
        return

    # 2. Caricamento modello
    model_path = os.path.join(model_dir, f"{clfName}.save")
    if not os.path.exists(model_path):
        print(f"Errore: modello non trovato in '{model_path}'")
        return
    model = pickle.load(open(model_path, 'rb'))

    print(f"========================================")
    print(f"REPORT VALUTAZIONE: {clfName.upper()}")
    print(f"========================================")
    print(f"Workflow rilevato: {'Preprocessor (Custom Pipeline)' if preprocessor is not None else 'Dati Raw'}")

    if clfName in ("rf", "rf_new"):
        print(f"Parametri RF:")
        print(f" - n_estimators: {model.n_estimators}")
        print(f" - max_features: {model.max_features}")
        print(f" - criterion:    {model.criterion}")
        print(f" - max_depth:    {model.max_depth}")
        print(f" - class_weight: {model.class_weight}")
        print(f" - min_samples_leaf: {model.min_samples_leaf}")
    elif clfName == "svm":
        m_iter = getattr(model, 'max_iter', 'Default')
        print(f"Parametri SVC: C={model.C}, kernel='{getattr(model, 'kernel', 'linear')}', max_iter={m_iter}")

    print(f"----------------------------------------\n")

    # 3. Trasformazione
    if preprocessor is not None:
        try:
            X_transformed = preprocessor.transform(X_val_raw)
        except Exception as e:
            print(f"ERRORE durante transform: {e}")
            print(f"Tipo di X passato al preprocessor: {type(X_val_raw)}")
            raise
    else:
        X_transformed = X_val_raw.values if hasattr(X_val_raw, "values") else X_val_raw

    # 4. Predizione
    y_pred = model.predict(X_transformed)

    # 5. Metriche
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_val_true, y_pred))
    print(f"Accuracy Score:          {accuracy_score(y_val_true, y_pred):.4f}")
    print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_val_true, y_pred):.4f}")

    # 6. Matrice di confusione
    cm = confusion_matrix(y_val_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'knn': 'Greens', 'svm': 'Blues', 'rf': 'Oranges'}
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=colors.get(clfName, 'Purples'), ax=ax, values_format='d')
    plt.title(f'Confusion Matrix - {clfName.upper()}')
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






def print_highly_correlated_numeric_features(df, threshold):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()

    # Triangolo superiore (esclude diagonale e duplicati)
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    correlations_dict = {}
    cols_to_drop = []

    for column in upper_tri.columns:
        high_corr = upper_tri[column][upper_tri[column] > threshold]

        if not high_corr.empty:
            cols_to_drop.append(column)
            correlations_dict[column] = [
                f"{idx} ({corr_matrix.loc[idx, column]:+.2f})"
                for idx in high_corr.index
            ]

    # ---- PRINT ----
    if not correlations_dict:
        print(f"Nessuna correlazione trovata sopra la soglia di {threshold}")
    else:
        print(f"--- Colonne con correlazione assoluta >= {threshold} ---")
        for col, matches in correlations_dict.items():
            print(f"{col} correla con: {', '.join(matches)}")

    return cols_to_drop






def round_features_to_int(df, features):
    """
    Arrotonda all'intero più vicino i valori delle feature specificate.
    
    Parametri:
    -----------
    df : pd.DataFrame
        Il DataFrame da processare
    features : list
        Lista di nomi delle colonne da arrotondare
    
    Ritorna:
    --------
    pd.DataFrame
        DataFrame con le feature arrotondate
    """
    # Crea una copia del DataFrame per non modificare l'originale
    df_rounded = df.copy()
    
    # Arrotonda ogni feature nella lista
    for feature in features:
        if feature in df_rounded.columns:
            df_rounded[feature] = np.round(df_rounded[feature]).astype('Int64')
        else:
            print(f"Warning: '{feature}' non trovata nel DataFrame")
    
    return df_rounded



def drop_constant_columns(df):
    """
    Rimuove le colonne che hanno un solo valore univoco in tutto il dataset.
    """
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    df.drop(columns=constant_cols, inplace=True)
    
    print("\n--- Report Colonne Costanti ---")
    if constant_cols:
        for col in constant_cols:
            print(f"Dropping colonna costante: {col}")
    else:
       print("Nessuna colonna monovalore trovata.")

    return