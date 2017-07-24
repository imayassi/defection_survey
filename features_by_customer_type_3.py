def customer_type_features(ct):
    if ct=='New':
        cont_features=['ABANDONED','RISK_FLAG_PY','ANC_PY', 'AGE_TAXPAYER',	'AGI',	'AMOUNT_INCOME_TAX_WITHHELD',	'AMOUNT_INCOME_TAX',	'AMOUNT_REFUND',	'AMOUNT_SALARIES_AND_WAGES',	'AMOUNT_STUDENT_LOAN_INTEREST_DEDUCTION',	'AMOUNT_TAX',	'AMOUNT_TOTAL_DEDUCTIONS',	'AMOUNT_TOTAL_PAYMENTS',	'AMOUNT_TOTAL_TAX',	'COST_PER_CUST_LAG7',	'COST_PER_CUST',	'FSCHA_FLAG',	'REFUND_TRANSFER_FLAG',	'REQUIRED_TAKE_FLAG',	'SESSIONS_TO_COMPLETE',	'STATE_ATTACH_COUNT',	'STUDENT_TAXPAYER',	'TOTAL_REVENUE','NUM_DEPENDENTS',	'NUM_EXEMPTIONS',	'NUM_W2']

        bool_features=['ABANDONED','RISK_FLAG_PY','ANC_PY',	'FSCHA_FLAG',	'REFUND_TRANSFER_FLAG',	'REQUIRED_TAKE_FLAG',	'STUDENT_TAXPAYER']
        catag_features=['COMPLETED_SKU', 'CUSTOMER_TYPE', 'DMA_AREA', 'FED_FORM_TYPE', 'FILING_STATUS',
                          'FIRST_COMPLETE_APP_TYPE', 'FIRST_COMPLETE_DEVICE_TYPE', 'IMPORT_TYPE', 'NEAUTH_DEVICE_TYPE', 'PRS_SCORE', 'REJECT_COUNT',
                          'START_DEVICE_TYPE', 'VAUTH_DEVICE_TYPE']

        cont_score_features = [ 'AGE_TAXPAYER','RISK_FLAG_PY','ANC_PY','AGI', 'AMOUNT_INCOME_TAX_WITHHELD', 'AMOUNT_INCOME_TAX',
                         'AMOUNT_REFUND', 'AMOUNT_SALARIES_AND_WAGES', 'AMOUNT_STUDENT_LOAN_INTEREST_DEDUCTION',
                         'AMOUNT_TAX', 'AMOUNT_TOTAL_DEDUCTIONS', 'AMOUNT_TOTAL_PAYMENTS', 'AMOUNT_TOTAL_TAX',
                         'COST_PER_CUST_LAG7', 'COST_PER_CUST', 'FSCHA_FLAG', 'REFUND_TRANSFER_FLAG',
                         'REQUIRED_TAKE_FLAG', 'SESSIONS_TO_COMPLETE', 'STATE_ATTACH_COUNT', 'STUDENT_TAXPAYER',
                         'TOTAL_REVENUE','NUM_DEPENDENTS',	'NUM_EXEMPTIONS',	'NUM_W2']

        bool_score_features = ['FSCHA_FLAG', 'RISK_FLAG_PY','ANC_PY','REFUND_TRANSFER_FLAG', 'REQUIRED_TAKE_FLAG', 'STUDENT_TAXPAYER']
        catag_score_features = ['COMPLETED_SKU', 'CUSTOMER_TYPE', 'DMA_AREA', 'FED_FORM_TYPE', 'FILING_STATUS',
                          'FIRST_COMPLETE_APP_TYPE', 'FIRST_COMPLETE_DEVICE_TYPE', 'IMPORT_TYPE', 'NEAUTH_DEVICE_TYPE', 'PRS_SCORE', 'REJECT_COUNT',
                          'START_DEVICE_TYPE', 'VAUTH_DEVICE_TYPE']

    return cont_features, bool_features, catag_features, cont_score_features, bool_score_features, catag_score_features