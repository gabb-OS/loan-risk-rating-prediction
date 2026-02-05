import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pickle

class ColumnDropper(BaseEstimator, TransformerMixin):
    """ Drop generic columns """
    def __init__(self, columns=[]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=[col for col in self.columns if col in X.columns])
    

class HighNanDropper(BaseEstimator, TransformerMixin):
    """ Rimuove colonne con alto numero di NaN"""
    
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.columns = []

    def fit(self, X, y=None):
        nan_ratio = X.isna().mean()
        self.columns = nan_ratio[nan_ratio > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=[col for col in self.columns if col in X.columns])
    

class HighlyCorrelatedDropper(BaseEstimator, TransformerMixin):
    """" Rimozione delle feature ridondanti identificate (High Correlation > threshold) """
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.columns = []

    def fit(self, X, y=None):
        numeric_df = X.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        # Selezioniamo il triangolo superiore della matrice; k=1 esclude la diagonale principale
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.columns = [column for column in upper_tri.columns if any(upper_tri[column] > self.threshold)]
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=[col for col in self.columns if col in X.columns])
    

class NumericExtractor(BaseEstimator, TransformerMixin):
    """ Estrae feature numeriche da stringhe (36/60 mesi, polishing anni di carriera...) """
    def __init__(self, columns=[]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        valid_cols = [col for col in self.columns if col in X.columns]
        for col in valid_cols:
            X[col] = (X[col].astype(str)
                        .replace('< 1', '0') 
                        .str.extract(r"(\d+)")
                        .astype(float))
        return X
    
    
class FeatureAverager(BaseEstimator, TransformerMixin):
    """Media di n colonne e rimuove gli originali"""
    def __init__(self, columns=[], new_name='new_avg_col'):
        self.columns = columns
        self.new_name = new_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Calcoliamo la media lungo l'asse delle righe (axis=1)
        valid_cols = [col for col in self.columns if col in X.columns]
        if valid_cols:
            X[self.new_name] = X[valid_cols].mean(axis=1)
        return X.drop(columns=[col for col in self.columns if col in X.columns])

    
class DateDifferenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_col, target_cols=None):
        self.reference_col = reference_col
        self.target_cols = target_cols or []
        self.date_format = '%b-%Y'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Check if reference exists
        if self.reference_col not in X.columns:
            return X # Skip if ref is missing (maybe dropped by NaN filter)

        ref_series = pd.to_datetime(X[self.reference_col], format=self.date_format, errors='coerce')
        
        # Ensure targets is a list and filter for existence
        targets = [self.target_cols] if isinstance(self.target_cols, str) else self.target_cols
        valid_targets = [col for col in targets if col in X.columns]
        
        cols_to_drop = []

        for col in valid_targets:
            target_series = pd.to_datetime(X[col], format=self.date_format, errors='coerce')
            new_col_name = f"months_since_{col}"

            X[new_col_name] = (
                (ref_series.dt.year - target_series.dt.year) * 12 +
                (ref_series.dt.month - target_series.dt.month)
            )
            X[new_col_name] = np.round(X[new_col_name]).astype('Int64')
            cols_to_drop.append(col)

        # Drop columns
        all_drops = cols_to_drop + [self.reference_col]
        # Final safety check before drop
        X = X.drop(columns=[c for c in all_drops if c in X.columns], errors='ignore')

        return X
    

class RoundToIntTransformer(BaseEstimator, TransformerMixin):
    """ Arrotonda le colonne selezionate all'intero piu vicino """
    def __init__(self, columns=[]):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in [c for c in self.columns if c in X.columns]:
                X[col] =  np.round(X[col]).astype('Int64')
        return X


## TabNet ##################################################################################################

class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_cols_ = None
    
    def fit(self, X, y=None):
        self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.categorical_cols_:
            if col in X.columns:
                X[col] = X[col].fillna("__NaN__")
        return X


class NumericalMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_cols_ = None
        self.imputer_ = None
    
    def fit(self, X, y=None):
        self.numerical_cols_ = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
        if self.numerical_cols_:
            self.imputer_ = SimpleImputer(strategy='median')
            self.imputer_.fit(X[self.numerical_cols_])
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.numerical_cols_ and self.imputer_:
            X[self.numerical_cols_] = self.imputer_.transform(X[self.numerical_cols_])
        return X


class CategoricalLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_cols_ = None
        self.label_encoders_ = {}
        self.cat_idxs_ = []
        self.cat_dims_ = []
        self.all_columns_ = None
    
    def fit(self, X, y=None):
        self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.all_columns_ = X.columns.tolist()
        
        for col in self.categorical_cols_:
            le = LabelEncoder()
            unique_vals = X[col].astype(str).unique().tolist()
            if "__UNKNOWN__" not in unique_vals:
                unique_vals.append("__UNKNOWN__")
            
            le.fit(unique_vals)
            self.label_encoders_[col] = le
        
        self.cat_idxs_ = []
        self.cat_dims_ = []
        for i, col in enumerate(self.all_columns_):
            if col in self.categorical_cols_:
                self.cat_idxs_.append(i)
                self.cat_dims_.append(len(self.label_encoders_[col].classes_))
        
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.categorical_cols_:
            if col in X.columns:
                le = self.label_encoders_[col]
                unknown_idx = np.where(le.classes_ == "__UNKNOWN__")[0][0]
                
                X[col] = X[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else unknown_idx
                )
        return X
    
    def get_cat_idxs(self):
        return self.cat_idxs_
    
    def get_cat_dims(self):
        return self.cat_dims_

class ToFloat32(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.values.astype(np.float32)


class CompletePipelineTabNet:
    
    def __init__(self, structure_pipeline, test_size=0.25, random_state=42):
        self.structure_pipeline = structure_pipeline
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize preprocessing steps
        self.cat_imputer = CategoricalImputer()
        self.num_imputer = NumericalMedianImputer()
        self.label_encoder = CategoricalLabelEncoder()
        self.to_float = ToFloat32()
        
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        X_transformed = self.structure_pipeline.fit_transform(X)
        
        self.cat_imputer.fit(X_transformed)
        X_transformed = self.cat_imputer.transform(X_transformed)
        
        self.num_imputer.fit(X_transformed)
        X_transformed = self.num_imputer.transform(X_transformed)
        
        self.label_encoder.fit(X_transformed)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        if not self.is_fitted_:
            raise ValueError("La Pipeline deve essere fittata prima del transform.")
        
        # Apply all transformations
        X_transformed = self.structure_pipeline.transform(X)
        X_transformed = self.cat_imputer.transform(X_transformed)
        X_transformed = self.num_imputer.transform(X_transformed)
        X_transformed = self.label_encoder.transform(X_transformed)
        X_transformed = self.to_float.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_cat_idxs(self):
        if not self.is_fitted_:
            raise ValueError("La Pipeline deve essere ancora fittata.")
        return self.label_encoder.get_cat_idxs()
    
    def get_cat_dims(self):
        if not self.is_fitted_:
            raise ValueError("La Pipeline deve essere ancora fittata.")
        return self.label_encoder.get_cat_dims()
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Pipeline salvata in {filepath}")
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"Pipeline caricata da {filepath}")
        return pipeline



## Functions ##################################################################################################
def remove_duplicates(df):
    print("\n Inizio rimozione duplicati...")

    # Individua se esistono colonne con lo stesso nome
    # Se esistono, allora se le colonne sono duplicati perfetti, droppiamo il duplicato
    # Se esistono, ma nono sono perfetti duplicati, per intervenire consciamente sarebbe necessario avere maggior domain knowledge
    feature_list = df.columns.to_list()
    has_duplicate_cols = len(feature_list) != len(set(feature_list))
    
    if has_duplicate_cols:
        df_undup = df.T.drop_duplicates().T
    else:
        df_undup = df

    # Rimuovi righe duplicate
    df_undup = df_undup.drop_duplicates()

    print("\n Fine rimozione duplicati.")
    return df_undup


































def drop_leakage_and_non_significant_cols(X):
    print("\n Inizio rimozione colonne leakage...")
    # "Settlement" indica una situazione avvenuta durante / dopo il prestito, non al momento della concessione
    settlement_data_leakage = [col for col in X.columns if 'settlement' in col]

    # "Hardship loans" sono concessioni per agevolare il pagamento di un prestito quando il debitore si trova in momenti di difficoltà economica (perdita lavoro, problemi medici, disastri naturali)
    # https://www.oaic.gov.au/privacy/your-privacy-rights/credit-reporting/hardship-assistance/what-is-a-financial-hardship-arrangement
    # https://financialrights.org.au/factsheet/financial-hardship/
    hardship_data_leakage = [col for col in X.columns if 'hardship' in col]

        # DROP OPERATIONS
    X.drop(columns=loan_performance_data_leakage, inplace=True)
    print("Nuovo # Colonne: " +  str(X.shape[1]) + "\n")

    X.drop(columns=settlement_data_leakage, inplace=True)
    print("Nuovo # Colonne: " +  str(X.shape[1]) + "\n")

    X.drop(columns=hardship_data_leakage, inplace=True)
    print("Nuovo # Colonne: " +  str(X.shape[1]) + "\n")

    X.drop(columns=other_leakage, inplace=True)
    print("Nuovo # Colonne: " +  str(X.shape[1]) + "\n")

    X.drop(columns=loan_contract_interest_rate, inplace=True)
    print("Nuovo # Colonne: " +  str(X.shape[1]) + "\n")

    X.drop(columns=other_non_significant, inplace=True)
    print("Nuovo # Colonne: " +  str(X.shape[1]) + "\n")

    print("\n Fine rimozione colonne leakage.")
    return


def drop_high_nan_cols(X):
    print("\n Inizio rimozione colonne con alto numero NaN...")

    X.drop(columns=high_nan_columns, inplace=True)
    print("Nuovo # Colonne: " +  str(X.shape[1]) + "\n")

    print("\n Fine rimozione colonne con alto numero NaN.")
    return


def drop_joint_and_secondary_cols(X):
    print("\n Inizio rimozione colonne joint_ e secondary_...")

    joint_and_secondary_cols = [col for col in X.columns if col.startswith('joint_') or col.startswith('secondary_')]

    X.drop(columns=joint_and_secondary_cols, inplace=True)
    print("Nuovo # Colonne: " +  str(X.shape[1]) + "\n")

    print("\n Fine rimozione colonne joint_ e secondary_")
    return


def drop_higly_correlated_numeric_features(X):
    print("\n Inizio rimozione colonne altamente correlate tra loro...")

    X.drop(columns=redundant_features, inplace=True)
    print("\nNuovo # Colonne: " +  str(X.shape[1]) + "\n")

    print("\n Fine rimozione colonne altamente correlate tra loro")
    return


def feature_extraction(X):
    print("\n Inizio feature extraction...")

    # Trasforma "36 months" e "60 months" in float type
    X['loan_contract_term_months'] = X['loan_contract_term_months'].str.extract(r'(\d+)').astype(float)

    # Strip della stringa "years"
    # Trasforma anni in float: < 1 diventa 0, 10+ diventa 10
    X['borrower_profile_employment_length'] = X['borrower_profile_employment_length'].str.replace(r'\+? years?', '', regex=True)
    X['borrower_profile_employment_length'] = X['borrower_profile_employment_length'].replace({ '< 1': 0}).astype(float)

    # FICO average da fico_score_low_bound e fico_score_high_bound
    X['fico_average'] = (X['fico_score_low_bound'] + X['fico_score_high_bound']) / 2
    X.drop(columns=['fico_score_low_bound', 'fico_score_high_bound'], inplace=True)
    print("\nNuovo # Colonne: " +  str(X.shape[1]) + "\n")

    # Loan issue date usato per calcolare una feature derivata: 
    # months_since_earliest_cr_line (Numero mesi passati tra prima richiesta credito e loan date )
    X['loan_issue_date'] = pd.to_datetime(X['loan_issue_date'], format='%b-%Y')
    X['credit_history_earliest_line'] = pd.to_datetime(X['credit_history_earliest_line'], format='%b-%Y')

    X['months_since_earliest_cr_line'] = (
        (X['loan_issue_date'].dt.year - X['credit_history_earliest_line'].dt.year) * 12 +
        (X['loan_issue_date'].dt.month - X['credit_history_earliest_line'].dt.month)
    )

    # Tutte le features rimanenti con date non sono rilevanti
    date_cols = [col for col in X.columns if 'date' in col]
    to_drop = [
        'credit_history_earliest_line',     # used for feature extraction
        ] + date_cols
    X.drop(columns=to_drop, inplace=True)
    print("Nuovo # Colonne: " +  str(X.shape[1]) + "\n")

    print("\n Fine feature extraction")
    return


def round_features_to_int(X):
    print("\n Inizio arrotondamento delle features di tipo Int...")

    for col in round_to_nearest_int:
        if col in X.columns:
            X[col] = np.round(X[col]).astype('Int64')
        else:
            print(f"Warning: '{col}' non trovata nel DataFrame")
    

    print("\n Fine arrotondamento delle features di tipo Int")
    return


def nan_management_general_fill(X):
    print("\n Inizio gestione NaN generici...")

    # Feature categoriche in cui i valori NaN sono riempiti con una nuova label Unknown
    X[categorical_to_unknown_cols] = X[categorical_to_unknown_cols].fillna('unknown')

    #"Months Since" columns (NaN = mai accaduto)
    # filliamo con grande numero (e.g., 100 months) per dire "molto tempo fa / mai"
    X[fill_big_cols] = X[fill_big_cols].fillna(100)

    # 2. FILL con 0 (se non e' presente, allora equivale a mai/nessuno -> 0)
    X[fill_zero_cols] = X[fill_zero_cols].fillna(0)

    print("\n Fine gestione NaN generici")
    return


def nan_management_imputation_fill(X):
    print("\n Inizio gestione NaN con imputation...")

    # Feature categoriche in cui i valori NaN possono essere riempiti con la moda:
    # valori con 2 etichette, in cui la mancanza di un dato viene trattato come "no" o come occorrenza piu' frequente
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_imputer.fit(X[fill_to_mode_cat])
    X[fill_to_mode_cat] = cat_imputer.transform(X[fill_to_mode_cat])

    num_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer.fit(X[fill_to_mode_num])
    X[fill_to_mode_num] = num_imputer.transform(X[fill_to_mode_num])


    # Feature da fillare con mediana
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X[numerical_to_median])
    X[numerical_to_median] = imputer.transform(X[numerical_to_median])

    print("\n Fine gestione NaN con imputation")
    return


def apply_capping(X):
    print("\n Inzio capping feature numeriche...")    
    lower_quantile = 0.01
    upper_quantile = 0.99
    # Selezioniamo solo le colonne numeriche
    numerical_cols = X.select_dtypes(include=['float', 'int']).columns

    for col in numerical_cols:
        lower_limit = X[col].quantile(lower_quantile)
        upper_limit = X[col].quantile(upper_quantile)

        # Applicazione del capping (clipping)
        X[col] = X[col].clip(lower=lower_limit, upper=upper_limit)

        print("\n Fine capping feature numeriche")

    return



def apply_log_transform_on_skewed_cols(X):
    print("\n Inzio log transformation...")   
    ###TODO
    
    for col in skewed_cols:
        # Verifica che non ci siano valori negativi prima di applicare il log
        if (X[col] >= 0).all():
            X[col] = np.log1p(X[col])
        else:
            print(f"Salto {col}: contiene valori negativi (impossibile applicare log).")
    
    print("\n Fine log transformation")

    return

