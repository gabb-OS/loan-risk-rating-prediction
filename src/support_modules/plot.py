import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_nan(df):
    """Grafico a barre orizzontali con % di NaN vs valori presenti per feature"""

    df_plot = pd.DataFrame({
        'Presenti (%)': (df.notna().mean() * 100),
        'Mancanti (NaN %)': (df.isna().mean() * 100)
    })

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
    """ Distribuzione di una feature con le percentuali """

    if not isinstance(df_feature, pd.Series):
        df_feature = pd.Series(df_feature)

    counts = df_feature.value_counts().sort_index()
    percentages = df_feature.value_counts(normalize=True).sort_index() * 100

    plt.figure(figsize=(8, 5))

    ax = counts.plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.8)

    labels = [f'{p:.1f}%' for p in percentages]
    ax.bar_label(ax.containers[0], labels=labels, padding=3, fontsize=10, fontweight='bold')

    plt.title(f'Distribuzione della feature: {feature_name}', fontsize=14, fontweight='bold')
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)

    plt.xticks(rotation=45 if len(str(counts.index[0])) > 3 else 0)
    plt.ylim(0, counts.max() * 1.15)

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_top_correlations_split(X, y, n=30):
    """
    Calcola e visualizza la correlazione tra X_train e y_train.
    """
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    y_numeric = y.map(grade_map)

    X_numeric = X.select_dtypes(include=[np.number])

    correlations = X_numeric.corrwith(y_numeric).dropna()

    top_n_idx = correlations.abs().sort_values(ascending=False).head(n).index
    top_features = correlations.loc[top_n_idx].sort_values(ascending=True)

    plt.figure(figsize=(8, 10))

    # rosso = correlazione positiva, blu = negativa
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_features]

    ax = top_features.plot(kind='barh', color=colors, alpha=0.8)

    for i, v in enumerate(top_features):
        ax.text(v + (0.01 if v > 0 else -0.06), i, f'{v:.2f}',
                va='center', fontsize=10, fontweight='bold')

    plt.title(f'Top {n} features correlate con il grade', fontsize=15, fontweight='bold')
    plt.xlabel('Coefficiente di Correlazione (Pearson)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
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

    """

    if save_plots and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Cartella '{output_folder}' creata/verificata.")

    numerical_cols = df.select_dtypes(include=['float, int']).columns

    print(f"Inizio generazione grafici per {len(df.columns)} feature...")

    for col in df.columns:

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"Analisi Feature Numerica: {col}", fontsize=16)

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

            sns.histplot(df[col], kde=True, ax=axes[1], color='skyblue')
            axes[1].set_title("Distribuzione (Histogram + KDE)")
            axes[1].set_xlabel(col)

            sns.boxplot(x=df[col], ax=axes[2], color='orange')
            axes[2].set_title("Outliers (Boxplot)")
            axes[2].set_xlabel(col)


            plt.tight_layout()

            if save_plots:
                safe_col_name = "".join([c if c.isalnum() else "_" for c in col])
                filename = os.path.join(output_folder, f"{safe_col_name}.png")
                plt.savefig(filename)
                plt.close(fig)
            else:
                plt.show()

    if save_plots:
        print(f"Tutti i grafici sono stati salvati in '{output_folder}'.")

def plot_leakage_evidence(x_series, y_series, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Genera un grafico combinato (Boxplot + Strip Plot) per visualizzare
    la correlazione/leakage tra una feature categorica e una numerica.
    """

    df_temp = pd.DataFrame({
        'x': x_series.values,
        'y': y_series.values
    })

    sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.linestyle": ":"})

    plt.figure(figsize=(12, 7))

    order = sorted(df_temp['x'].unique())
    my_palette = sns.color_palette("viridis", len(order))

    # strip plot sullo sfondo, boxplot in primo piano (zorder)
    sns.stripplot(
        data=df_temp, x='x', y='y', order=order,
        color='grey', size=2.5, alpha=0.50, jitter=0.3, zorder=0
    )

    sns.boxplot(
        data=df_temp, x='x', y='y', order=order,
        palette=my_palette, showfliers=False, width=0.5, linewidth=2,
        boxprops=dict(alpha=0.9), zorder=10
    )

    final_title = title if title else f"Distribuzione: {x_series.name} vs {y_series.name}"
    final_xlabel = xlabel if xlabel else x_series.name
    final_ylabel = ylabel if ylabel else y_series.name

    plt.title(final_title, fontsize=15, fontweight='bold', pad=20)
    plt.xlabel(final_xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(final_ylabel, fontsize=12, fontweight='bold')

    sns.despine(trim=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Grafico salvato in: {save_path}")

    plt.show()
