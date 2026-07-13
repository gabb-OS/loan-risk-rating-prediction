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

def evaluate_model(X_val_raw, y_val_true, clfName, model_dir="models", preproc_dir="preprocessing"):
    """
    Valuta il modello applicando ESATTAMENTE la stessa pipeline di preprocessing
    usata in training (stessa logica di preprocess()).
    """

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

    if preprocessor is not None:
        try:
            X_transformed = preprocessor.transform(X_val_raw)
        except Exception as e:
            print(f"ERRORE durante transform: {e}")
            print(f"Tipo di X passato al preprocessor: {type(X_val_raw)}")
            raise
    else:
        X_transformed = X_val_raw.values if hasattr(X_val_raw, "values") else X_val_raw

    y_pred = model.predict(X_transformed)

    print("CLASSIFICATION REPORT:")
    print(classification_report(y_val_true, y_pred))
    print(f"Accuracy Score:          {accuracy_score(y_val_true, y_pred):.4f}")
    print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_val_true, y_pred):.4f}")

    cm = confusion_matrix(y_val_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'knn': 'Greens', 'svm': 'Blues', 'rf': 'Oranges'}
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=colors.get(clfName, 'Purples'), ax=ax, values_format='d')
    plt.title(f'Confusion Matrix - {clfName.upper()}')
    plt.show()

def print_highly_correlated_numeric_features(df, threshold):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()

    # triangolo superiore (esclude diagonale e duplicati)
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
    """
    df_rounded = df.copy()

    for feature in features:
        if feature in df_rounded.columns:
            df_rounded[feature] = np.round(df_rounded[feature]).astype('Int64')
        else:
            print(f"Warning: '{feature}' non trovata nel DataFrame")

    return df_rounded