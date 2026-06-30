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

# Il tasso di interesse di un prestito è calcolato basandosi sul Grading assegnato al prestito stesso.
# Essendo una conseguenza del nostro target "grade", è da considerarsi data leakage
# https://www.airtel.in/blog/personal-loan/how-does-loan-grading-work/ (Accessed 02/02/2026)
loan_contract_interest_rate = [
    'loan_contract_interest_rate'
]

# "Settlement" indica una situazione avvenuta durante / dopo il prestito, non al momento della concessione
settlement_data_leakage = [
    'debt_settlement_flag_indicator',
    'settlement_status_label',
    'settlement_amount_total', 
    'settlement_percentage', 
    'settlement_term_months'
]

# "Hardship loans" sono concessioni per agevolare il pagamento di un prestito quando il debitore si trova in momenti di difficoltà economica (perdita lavoro, problemi medici, disastri naturali)
# https://www.oaic.gov.au/privacy/your-privacy-rights/credit-reporting/hardship-assistance/what-is-a-financial-hardship-arrangement
# https://financialrights.org.au/factsheet/financial-hardship/
hardship_data_leakage = [
    'hardship_flag_indicator',
    'hardship_type_label',
    'hardship_reason_label',
    'hardship_status_label',
    'hardship_deferral_term_months',
    'hardship_amount_total',
    'hardship_duration_days',
    'hardship_days_past_due',
    'hardship_loan_status_label',
    'hardship_payoff_balance',
    'hardship_last_payment_amount_total'
]

other_leakage = [
    'original_projected_additional_accrued_interest',           # interesse addizionale previsto, presumibilmente in seguito a modifiche di piani ammortamento o hardship
    #'loan_issue_date',                                         # Il grade è influenzato dalla situazione creditizia del richiedente, più che dal periodo
                                                                # droppato in un secondo momento, dopo averlo usato per feature extraction
]

other_non_significant = [
    'investor_side_funded_amount',
    'loan_portfolio_total_funded',
    'platform_policy_code_id',                                      # id interno al prestatore
    'loan_title',                                                   # non significant column, grande sparsita' di dati. Sufficiente loan_purpose_category come aggregazione di scopo del prestito
    'borrower_address_zip',                                         # non significant column, esiste una colonna per identificazione stati
]

####################################################################################################
joint_and_secondary_cols = [
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

####################################################################################################
number_from_string_cols = [
    'loan_contract_term_months',            # 36 mesi / 60 mesi
    'borrower_profile_employment_length',   # < 1 years, 10+ years
]

average_cols = [
    'fico_score_low_bound',
    'fico_score_high_bound'
]

date_diff_reference = 'loan_issue_date'
date_diff_target = [ 'credit_history_earliest_line' ]

####################################################################################################

# Lista delle feature che richiedono valori interi: conteggi di linee di credito e conti, mesi, fico score
round_to_nearest_int = [
    # Months
    'loan_contract_term_months',
    'months_since_last_delinquency',
    'months_since_last_public_record',
    'months_since_last_major_derog',
    'months_since_recent_installment_loan',
    'months_since_oldest_installment_acct',
    'months_since_oldest_revolving_acct',
    'months_since_recent_revolving_acct',
    'months_since_recent_trade_line',
    'months_since_recent_bankcard',
    'months_since_recent_bankcard_delinquency',
    'months_since_recent_inquiry',
    'months_since_recent_revolving_delinquency',
    'months_since_credit_history_earliest_line',

    # account, eventi, inquiry, delinquency
    'credit_delinquencies_2yrs',
    'credit_inquiries_6m',
    'credit_open_accounts',
    'credit_public_records',
    'collections_12m_ex_med',
    'accounts_now_delinquent',
    'open_accounts_6m',
    'open_active_installment_loans',
    'open_installment_loans_12m',
    'open_installment_loans_24m',
    'open_revolving_accounts_12m',
    'open_revolving_accounts_24m',
    'finance_inquiries',
    'credit_union_trades_total',
    'credit_inquiries_12m',
    'accounts_open_past_24m',
    'chargeoffs_within_12m',
    'mortgage_accounts',
    'accounts_ever_120dpd',
    'active_bankcard_tradelines',
    'active_revolving_tradelines',
    'bankcard_satisfactory_accounts',
    'bankcard_tradelines',
    'installment_tradelines',
    'open_revolving_tradelines',
    'revolving_accounts',
    'tradelines_120dpd_2m',
    'tradelines_30dpd',
    'tradelines_90dpd_24m',
    'tradelines_open_past_12m',
    'public_record_bankruptcies',
    'tax_liens_total',

    'borrower_profile_employment_length',
    'fico_average'

]

####################################################################################################

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
    'open_revolving_accounts_12m', 'open_revolving_accounts_24m',
    'finance_inquiries', 'credit_inquiries_12m', 'credit_union_trades_total',
    'open_active_installment_loans', 'tradelines_open_past_12m', 'accounts_open_past_24m',
    'open_revolving_tradelines', 'active_bankcard_tradelines', 'installment_tradelines',
    'revolving_accounts', 'bankcard_satisfactory_accounts', 'active_revolving_tradelines', 
    'bankcard_tradelines', 'credit_open_accounts', 'credit_total_accounts',
    'tradelines_120dpd_2m', 'tradelines_30dpd', 'tradelines_90dpd_24m',
    'accounts_ever_120dpd', 'chargeoffs_within_12m', 'collections_12m_ex_med',
    'accounts_now_delinquent', 'public_record_bankruptcies', 'credit_public_records',
    'credit_delinquencies_2yrs', 'tax_liens_total', 'borrower_profile_employment_length'
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