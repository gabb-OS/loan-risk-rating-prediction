import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

####################################################################################################

loan_performance_data_leakage = [
    'loan_status_current_code',                         # prestito in regola, in ritardo, totalmente pagato...
    'outstanding_principal_balance',                    # "outstanding principal" e' la parte del capitale da restituire
    'outstanding_principal_investor_side',              # similmente
    'total_payment_received',                           # somma pagata al creditore
    'total_payment_investor_side',
    'total_received_principal',                         # somma pagata al creditore che copre la il capitale del prestito
    'total_received_interest',                          # ... copre gli interessi
    'total_received_late_fees',                         # ... copre le penali
    'recoveries_cash',                                  # somma recuperata dopo un prestito andato in default
    'collection_recovery_fee',                          # spese per il recupero crediti
    'last_payment_date',                                # data ultimo pagamento effettuato
    'last_payment',                                     # importo ultimo pagamento
    'next_payment_date',                                # data prossimo pagamento
    'last_credit_pull_date',                            # data ultimo check profilo creditizio, durante il periodo di prestito
    'last_fico_score_high_bound',                       # ultimo punteggio FICO rilevato: al momento della concessione del prestito si usa 'fico_score_low_bound', 'fico_score_high_bound
    'last_fico_score_low_bound',
    'total_collection_amount',
    'loan_payment_installments_count'                   # potrebbe semprare il numero di rate, ma la tipologia di valori contenuti fa pensare al valore economico della singola rata (calcolo derivante di interest rate)
]
other_leakage = [
    'original_projected_additional_accrued_interest',           # interesse addizionale previsto, presumibilmente in seguito a modifiche di piani ammortamento o hardship
    #'loan_issue_date',                                         # Il grade è influenzato dalla situazione creditizia del richiedente, più che dal periodo
                                                                # droppato in un secondo momento, dopo averlo usato per feature extraction
    'investor_side_funded_amount',
    'loan_portfolio_total_funded',
]
# Il tasso di interesse di un prestito è calcolato basandosi sul Grading assegnato al prestito stesso.
# Essendo una conseguenza del nostro target "grade", è da considerarsi data leakage
# https://www.airtel.in/blog/personal-loan/how-does-loan-grading-work/ (Accessed 02/02/2026)
loan_contract_interest_rate = [
    'loan_contract_interest_rate'
]

other_non_significant = [
    'platform_policy_code_id',                                      # id interno al prestatore
    'loan_title',                                                   # non significant column, grande sparsita' di dati. Sufficiente loan_purpose_category come aggregazione di scopo del prestito
    'borrower_address_zip',                                         # non significant column, esiste una colonna per identificazione stati
]

high_nan_columns = [
    'joint_income_annual',
    'joint_dti_ratio', 
    'joint_income_verification_status', 
    'joint_revolving_balance',
    'secondary_applicant_fico_low',
    'secondary_applicant_fico_high',
    'secondary_applicant_earliest_credit_line',
    'secondary_applicant_inquiries_6m',
    'secondary_applicant_mortgage_accounts',
    'secondary_applicant_open_accounts',
    'secondary_applicant_revolving_utilization',
    'secondary_applicant_open_active_installment_loans',
    'secondary_applicant_revolving_accounts',
    'secondary_applicant_chargeoffs_12m',
    'secondary_applicant_collections_12m_ex_med',
    'secondary_applicant_months_since_last_major_derog'
    ]

round_to_nearest_int = [
    'loan_contract_term_months', 'credit_delinquencies_2yrs', 'credit_inquiries_6m',
    'months_since_last_delinquency', 'months_since_last_public_record', 
    'credit_open_accounts', 'credit_public_records', 'credit_total_accounts',
    'collections_12m_ex_med', 'months_since_last_major_derog', 'accounts_now_delinquent',
    'open_accounts_6m', 'open_active_installment_loans', 'open_installment_loans_12m',
    'open_installment_loans_24m', 'months_since_recent_installment_loan', 
    'open_revolving_accounts_12m', 'open_revolving_accounts_24m', 'finance_inquiries',
    'credit_union_trades_total', 'credit_inquiries_12m', 'accounts_open_past_24m',
    'chargeoffs_within_12m', 'months_since_oldest_installment_acct', 
    'months_since_oldest_revolving_acct', 'months_since_recent_revolving_acct',
    'months_since_recent_trade_line', 'mortgage_accounts', 'months_since_recent_bankcard',
    'months_since_recent_bankcard_delinquency', 'months_since_recent_inquiry',
    'months_since_recent_revolving_delinquency', 'accounts_ever_120dpd',
    'active_bankcard_tradelines', 'active_revolving_tradelines', 
    'bankcard_satisfactory_accounts', 'bankcard_tradelines', 'installment_tradelines',
    'open_revolving_tradelines', 'revolving_accounts', 'tradelines_120dpd_2m',
    'tradelines_30dpd', 'tradelines_90dpd_24m', 'tradelines_open_past_12m',
    'public_record_bankruptcies', 'tax_liens_total', 'fico_average', 
    'months_since_earliest_cr_line'
]
categorical_to_unknown_cols = [
  'borrower_address_state',
  'loan_purpose_category',
  'borrower_income_verification_status',
  'borrower_housing_ownership_status'
]
fill_big_cols = [
    'months_since_last_public_record',
    'months_since_recent_bankcard_delinquency',
    'months_since_last_major_derog',
    'months_since_recent_revolving_delinquency',
    'months_since_last_delinquency', 
    'months_since_recent_inquiry', 
    'months_since_recent_bankcard',
    'months_since_recent_trade_line'
]
fill_zero_cols = [
    'open_accounts_6m', 'open_installment_loans_12m', 'open_installment_loans_24m',
    'open_revolving_accounts_12m', 'open_revolving_accounts_24m', 'finance_inquiries',
    'credit_inquiries_12m', 'delinquency_amount', 
    'tax_liens_total', 'mortgage_accounts', 'chargeoffs_within_12m', 
    'collections_12m_ex_med', 'accounts_now_delinquent', 'public_record_bankruptcies',
    'credit_public_records', 'credit_delinquencies_2yrs',
    'borrower_profile_employment_length',

    'tradelines_120dpd_2m', 'tradelines_30dpd', 'tradelines_90dpd_24m', 
    'accounts_ever_120dpd', 'credit_union_trades_total', 'open_active_installment_loans', 
    'credit_inquiries_6m', 'tradelines_open_past_12m', 'accounts_open_past_24m',
    'open_revolving_tradelines', 'active_bankcard_tradelines', 'installment_tradelines',
    'revolving_accounts', 'bankcard_satisfactory_accounts',
    'active_revolving_tradelines',  'bankcard_tradelines',
    'credit_open_accounts', 'credit_total_accounts'
]

redundant_features = [
    'satisfactory_accounts',
    'total_high_credit_limit',
    'total_installment_high_credit_limit',
    'revolving_tradelines_balance_gt_0',
]

fill_to_mode_cat = [
    'disbursement_method_type',
    'application_type_label',
    'listing_initial_status',
    'loan_payment_plan_flag'         # n/y
]

fill_to_mode_num = [
    'loan_contract_term_months'    # 36/60 mesi
]

numerical_to_median = [
    'borrower_profile_employment_length', 'fico_average', 'borrower_income_annual', 
    'borrower_dti_ratio', 'loan_contract_approved_amount', 'average_current_balance', 
    'revolving_balance', 'total_current_balance', 'total_bankcard_credit_limit', 
    'total_revolving_high_credit_limit', 'bankcard_open_to_buy', 'bankcard_max_balance', 
    'total_balance_installment_loans', 'months_since_earliest_cr_line', 
    'months_since_oldest_installment_acct', 'months_since_oldest_revolving_acct', 
    'months_since_recent_revolving_acct', 'revolving_utilization', 'bankcard_utilization', 
    'overall_utilization', 'installment_utilization', 'bankcard_util_gt_75_ratio', 
    'tradelines_never_delinquent_ratio', 'total_balance_ex_mortgage', 'months_since_recent_installment_loan',
]

skewed_cols = [

]

####################################################################################################

def remove_duplicates(df):
    print("\n Inizio rimozione duplicati...")

    # Individua se esistono colonne con lo stesso nome
    # Se esistono, allora se le colonne sono duplicati perfetti, droppiamo il duplicato
    # Se esistono, ma nono sono perfetti duplicati, per intervenire consciamente sarebbe necessario avere maggior domain knowledge
    feature_list = df.columns.to_list()
    has_duplicate_cols = len(feature_list) != len(set(feature_list))
    if has_duplicate_cols:
        df.T.drop_duplicates(inplace=True).T

    # Rimuovi righe duplicate
    df.drop_duplicates(inplace=True)

    print("\n Inizio rimozione duplicati.")
    return


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

