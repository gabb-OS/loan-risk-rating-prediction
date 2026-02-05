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
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        for col in [c for c in self.columns if c in X.columns]:
                X[col] =  np.round(X[col]).astype('Int64')
        return X
    

# ============================================================================== 
#  K-Nearest Neighbors (KNN) - SVC
# ==============================================================================
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.limits_ = {}

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.limits_[col] = (
                X[col].quantile(self.lower_quantile),
                X[col].quantile(self.upper_quantile)
            )
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.limits_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X

class SkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.skewed_cols_ = []

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        skew_values = X[numeric_cols].skew().abs()
        
        self.skewed_cols_ = skew_values[skew_values > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.skewed_cols_:
            if col in X.columns:
                if (X[col] >= 0).all():
                    X[col] = np.log1p(X[col]) 
        return X


# ============================================================================== 
# TabNet
# ==============================================================================
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
    
    def __init__(self, structure_pipeline, random_state=42):
        self.structure_pipeline = structure_pipeline
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



# ============================================================================== 
# Support Functions
# ==============================================================================
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