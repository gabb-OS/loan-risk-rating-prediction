import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 


def plot_nan(df):
    """Plot stacked barh with % of NaN vs present values per feature"""

    df_plot = pd.DataFrame({
        'Presenti (%)': (df.notna().mean() * 100),
        'Mancanti (NaN %)': (df.isna().mean() * 100)
    })

    # Ordiniamo per % di NaN crescente
    df_plot = df_plot.sort_values(by='Mancanti (NaN %)', ascending=True)

    ax = df_plot[['Presenti (%)', 'Mancanti (NaN %)']].plot(
        kind='barh',
        stacked=True,
        color=['#2ca02c', '#d62728'],
        figsize=(10, 30)
    )

    plt.title('Percentuale Valori Presenti vs NaN per Feature (Ordinato)', fontsize=16)
    plt.ylabel('Features', fontsize=14)
    plt.xlabel('Percentuale (%)', fontsize=12)

    plt.yticks(fontsize=7)
    plt.xlim(0, 100)

    plt.legend(loc='lower right', title='Stato')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('missing_values_horizontal_sorted_pct.png')
    plt.show()


def plot_feature_distribution(df_feature, feature_name):
    """ Plot distribution of a given feature with percentages on top of bars """

    # --- FIX: Convertiamo sempre in Pandas Series per sicurezza ---
    if not isinstance(df_feature, pd.Series):
        df_feature = pd.Series(df_feature)

    # 1. Calcolo frequenze e percentuali
    counts = df_feature.value_counts().sort_index()
    percentages = df_feature.value_counts(normalize=True).sort_index() * 100

    plt.figure(figsize=(8, 5))

    # Creazione del grafico (catturiamo l'oggetto 'ax' per aggiungere le etichette)
    ax = counts.plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.8)

    # 2. Aggiunta delle percentuali sopra le barre
    # Generiamo le etichette formattate (es. "15.4%")
    labels = [f'{p:.1f}%' for p in percentages]

    # bar_label aggiunge automaticamente il testo sopra ogni barra
    ax.bar_label(ax.containers[0], labels=labels, padding=3, fontsize=10, fontweight='bold')

    # 3. Formattazione estetica
    plt.title(f'Distribuzione della feature: {feature_name}', fontsize=14, fontweight='bold')
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)

    # Rotazione delle labels X se sono stringhe lunghe, altrimenti 0
    plt.xticks(rotation=45 if len(str(counts.index[0])) > 3 else 0)

    # Espandiamo il limite superiore dell'asse Y per far stare le scritte
    plt.ylim(0, counts.max() * 1.15)

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_outliers_analyze_boxplot(df):
    # 1. Select only numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("No numerical columns found!")
        return

    # 2. Prepare the figure
    # Create one subplot per column
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 4 * len(numeric_cols)))

    # Handle case if there's only one numerical column (axes is not a list)
    if len(numeric_cols) == 1:
        axes = [axes]

    print(f"{'Column':<20} | {'Lower Cutoff':<15} | {'Upper Cutoff':<15}")
    print("-" * 55)

    for i, col in enumerate(numeric_cols):
        ax = axes[i]

        # 3. Calculate IQR stats
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"{col:<20} | {lower_bound:<15.2f} | {upper_bound:<15.2f}")

        # 4. Plot Boxplot
        # vert=False makes it horizontal, which is often easier to read
        ax.boxplot(df[col].dropna(), vert=False, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='blue'),
                   medianprops=dict(color='red', linewidth=2))

        # 5. Add Cutoff Lines
        ax.axvline(lower_bound, color='orange', linestyle='--', linewidth=1.5, label='Lower Limit')
        ax.axvline(upper_bound, color='orange', linestyle='--', linewidth=1.5, label='Upper Limit')

        ax.set_title(f"Distribution & Cutoffs: {col}")
        ax.set_xlabel("Value")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_outliers_scatter(df):
    """
    Genera automaticamente i plot per ogni feature del DataFrame.
    - Scatter plot per variabili numeriche (con linee di media e deviazione standard).
    - Bar chart per variabili categoriche.
    """
    # Selezione automatica delle colonne per tipo
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in df.columns:
        plt.figure(figsize=(10, 4))

        if col in numerical_cols:
            # --- NUMERICAL: SCATTER PLOT ---
            plt.scatter(x=df.index, y=df[col], alpha=0.5)

            # Calcolo statistiche sul dataframe di input
            mean = df[col].mean()
            std = df[col].std()
            
            # Linee guida per identificare gli outlier
            plt.axhline(mean + 3*std, color='r', linestyle='--', label='Media + 3SD')
            plt.axhline(mean - 3*std, color='r', linestyle='--', label='Media - 3SD')
            plt.axhline(mean, color='g', linestyle='-', label='Media')
            
            plt.legend()
            plt.title(f"Distribuzione Numerica: {col}")
            plt.ylabel("Valore")
            plt.xlabel("Indice")

        else:
            # --- CATEGORICAL: BAR CHART ---
            df[col].value_counts().plot(kind='bar', color='orange')
            plt.title(f"Frequenze Categoriche: {col}")
            plt.ylabel("Conteggio")
            plt.xlabel("Categoria")
            plt.xticks(rotation=45, ha='right')

        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

def plot_top_correlations_split(X, y, n=50):
    """
    Calcola e visualizza la correlazione tra X_train e y_train.

    Parameters:
    X (pd.DataFrame): Il set delle feature (X_train).
    y (pd.Series): Il target (y_train) contenente i gradi 'A', 'B', ecc.
    """
    # 1. Mappatura del Target in numerico
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    y_numeric = y.map(grade_map)

    # 2. Selezione delle sole feature numeriche da X
    X_numeric = X.select_dtypes(include=[np.number])

    # 3. Calcolo della correlazione di ogni feature con il target numerico
    # corrwith restituisce una serie con la correlazione per ogni colonna
    correlations = X_numeric.corrwith(y_numeric).dropna()

    # 4. Selezione delle top N feature (per valore assoluto)
    top_n_idx = correlations.abs().sort_values(ascending=False).head(n).index
    top_features = correlations.loc[top_n_idx].sort_values(ascending=True)

    # 5. Plotting
    plt.figure(figsize=(10, 12))

    # Colore rosso per corr. positiva, blu per negativa
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_features]

    ax = top_features.plot(kind='barh', color=colors, alpha=0.8)

    # Aggiunta dei valori numerici accanto alle barre
    for i, v in enumerate(top_features):
        ax.text(v + (0.01 if v > 0 else -0.06), i, f'{v:.2f}',
                va='center', fontsize=10, fontweight='bold')

    plt.title(f'Top {n} Feature correlate con il Grade (X_train vs y_train)', fontsize=15, fontweight='bold')
    plt.xlabel('Coefficiente di Correlazione (Pearson)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


def visualize_rf_tree(model_input, X_train, y_train, max_depth=3, tree_index=0):
    """
    Visualizza un singolo albero da un modello Random Forest.
    model_input: può essere il percorso del file .save o l'oggetto modello/pipeline.
    """
    try:
        # 1. Caricamento del modello se viene passato un percorso
        if isinstance(model_input, str):
            if not os.path.exists(model_input):
                print(f"Errore: Il file '{model_input}' non esiste.")
                return
            with open(model_input, "rb") as f:
                model = pickle.load(f)
        else:
            model = model_input

        # 2. Estrazione del classificatore se è dentro una Pipeline
        # GridSearchCV o Pipeline spesso avvolgono il classificatore sotto 'clf'
        if hasattr(model, 'named_steps'):
            clf = model.named_steps.get('clf', model)
        elif hasattr(model, 'best_estimator_'):
            clf = model_input.best_estimator_
            if hasattr(clf, 'named_steps'): clf = clf.named_steps.get('clf', clf)
        else:
            clf = model

        # 3. Verifica se è una Random Forest e ha gli alberi (estimators_)
        if not hasattr(clf, 'estimators_'):
            print("Errore: Il modello fornito non sembra essere una Random Forest o non è ancora addestrato.")
            return

        # 4. Preparazione nomi feature e classi
        feature_names = X_train.columns.tolist()
        class_names = [str(c) for c in sorted(np.unique(y_train))]

        # 5. Selezione dell'albero specifico
        tree_to_plot = clf.estimators_[tree_index]

        # 6. Plotting
        plt.figure(figsize=(25, 12))
        plot_tree(tree_to_plot,
                  max_depth=max_depth,
                  feature_names=feature_names,
                  class_names=class_names,
                  filled=True,
                  rounded=True,
                  proportion=True,
                  fontsize=10)

        plt.title(f"Albero n. {tree_index} della Random Forest (Profondità visualizzata: {max_depth})", fontsize=16)
        plt.show()

    except Exception as e:
        print(f"Errore durante la visualizzazione dell'albero: {e}")


def plot_knn_error_rate(grid_search_obj, param_name='n_neighbors'):
    """
    Visualizza l'andamento dello score al variare di un parametro (es. n_neighbors)
    basandosi sui risultati cv_results_ di GridSearchCV.
    """
    results = grid_search_obj.cv_results_
    
    # Trova la chiave corretta nei parametri (es. 'param_knn__n_neighbors' o 'param_n_neighbors')
    param_key = next((k for k in results.keys() if param_name in k and 'param_' in k), None)
    
    if not param_key:
        print(f"Parametro {param_name} non trovato nei risultati della GridSearch.")
        return

    means = results['mean_test_score']
    params = results[param_key].data.astype(int) # Assicura che siano interi per il plot

    plt.figure(figsize=(10, 6))
    plt.plot(params, means, linestyle='--', marker='o', color='blue')
    plt.title(f'Andamento Accuracy vs {param_name}')
    plt.xlabel(f'{param_name} (K)')
    plt.ylabel('Mean CV Accuracy')
    plt.grid(True)
    plt.show()



def analyze_feature_distributions(df, save_plots=False, output_folder="plots_analysis"):
    """
    Genera una dashboard di analisi per ogni feature del DataFrame.
    
    Per le variabili NUMERICHE (continue) genera 3 grafici affiancati:
      1. Scatter plot con linee di media e +/- 3 Deviazioni Standard (Outlier detection base).
      2. Istogramma con KDE (Distribuzione).
      3. Boxplot (Outlier detection statistica).
      
    Per le variabili CATEGORICHE (o numeriche con bassa cardinalità) genera:
      1. Bar chart delle frequenze.

    Args:
        df (pd.DataFrame): Il dataframe di input.
        save_plots (bool): Se True, salva i grafici in una cartella invece di mostrarli.
        output_folder (str): Nome della cartella di output (creata se non esiste).
    """
    
    # Crea la cartella se necessario
    if save_plots and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Cartella '{output_folder}' creata/verificata.")

    # Identifica le colonne numeriche
    numerical_cols = df.select_dtypes(include=['float, int']).columns
    
    print(f"Inizio generazione grafici per {len(df.columns)} feature...")

    for col in df.columns:
        # Euristica: se è numerica ma ha meno di 20 valori unici (es. flag 0/1, rating 1-5),
        # trattala come categorica per il grafico (meglio un bar chart che un boxplot).

            # --- LAYOUT A 3 COLONNE PER VARIABILI CONTINUE ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Analisi Feature Numerica: {col}", fontsize=16)

        # 1. SCATTER PLOT (Dalla funzione 1)
        axes[0].scatter(x=df.index, y=df[col], alpha=0.5, s=10, color='steelblue')
        
        mean = df[col].mean()
        std = df[col].std()
        
        axes[0].axhline(mean + 3*std, color='r', linestyle='--', label='Media + 3SD')
        axes[0].axhline(mean - 3*std, color='r', linestyle='--', label='Media - 3SD')
        axes[0].axhline(mean, color='g', linestyle='-', label='Media')
        axes[0].legend()
        axes[0].set_title("Scatter: Index vs Value")
        axes[0].set_ylabel("Valore")
        axes[0].set_xlabel("Indice")

        # 2. HISTOGRAMMA + KDE (Dalla funzione 2)
        sns.histplot(df[col], kde=True, ax=axes[1], color='skyblue')
        axes[1].set_title("Distribuzione (Histogram + KDE)")
        axes[1].set_xlabel(col)

        # 3. BOXPLOT (Dalla funzione 2)
        sns.boxplot(x=df[col], ax=axes[2], color='orange')
        axes[2].set_title("Outliers (Boxplot)")
        axes[2].set_xlabel(col)


        plt.tight_layout()

        # Logica di salvataggio o visualizzazione
        if save_plots:
            # Pulisci il nome del file da caratteri illegali
            safe_col_name = "".join([c if c.isalnum() else "_" for c in col])
            filename = os.path.join(output_folder, f"{safe_col_name}.png")
            plt.savefig(filename)
            plt.close(fig) # Chiude la figura per liberare memoria RAM
        else:
            plt.show()

    if save_plots:
        print(f"Tutti i grafici sono stati salvati in '{output_folder}'.")


def plot_feature_importance(model_input, feature_names, top_n=20, title="Top Feature Importance"):
    """
    Estrae e visualizza le feature più importanti di una Random Forest.
    """
    model = None

    # 1. Caricamento (da file o oggetto)
    if isinstance(model_input, str):
        with open(model_input, "rb") as f:
            model = pickle.load(f)
    else:
        model = model_input

    # 2. Gestione Pipeline (estraiamo il classificatore 'clf')
    if hasattr(model, 'named_steps'):
        clf = model.named_steps.get('clf', model.steps[-1][1])
    else:
        clf = model

    # 3. Estrazione importanze
    if not hasattr(clf, 'feature_importances_'):
        print("Errore: Il modello non supporta feature_importances_ (non è un modello basato su alberi).")
        return

    importances = clf.feature_importances_

    # Creiamo un DataFrame per facilitare l'ordinamento
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # 4. Plotting
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=fi_df.head(top_n), palette='viridis')

    plt.title(f"{title} (Top {top_n})", fontsize=15)
    plt.xlabel("Importanza (Gini Impurity Decrease)")
    plt.ylabel("Variabili")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

    return fi_df