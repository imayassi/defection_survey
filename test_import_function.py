import time

start_time = time.time()
import pandas as pd
from sklearn.decomposition import PCA, NMF
from binning import bin
import numpy as np
from statsmodels.tools import categorical
import skfuzzy as fuzzfcr
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn import datasets, cluster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score, roc_auc_score,  average_precision_score, f1_score
from sklearn.cross_decomposition import PLSCanonical
from sklearn import linear_model, decomposition, datasets
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from binning import bin
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.naive_bayes import BernoulliNB
from sklearn.utils import check_random_state
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
# import vertica_python
import pyodbc
import pandas as pd
from pandas import Series,DataFrame
from features_by_customer_type_3 import customer_type_features
from sklearn.feature_selection import SelectKBest, chi2
conn = pyodbc.connect(dsn='VerticaProd')

defection=pd.DataFrame()
random_state = np.random.RandomState(0)
i='False'
j='False'
k='False'
ct='New'

cont_features,bool_features,catag_features,cont_score_features,bool_score_features,catag_score_features=customer_type_features(ct)


# cont_features=[	'ABANDONED',	'AGE_DEPENDENT_MAX',	'AGE_DEPENDENT_MAX_PY',	'AGE_DEPENDENT_MAX_PY2',	'AGE_DEPENDENT_MIN_PY',	'AGE_DEPENDENT_MIN_PY2',	'AGE_SPOUSE_PY2',	'AGE_TAXPAYER_PY',	'AGE_TAXPAYER_PY2',	'AMOUNT_ADJUSTMENTS',	'AMOUNT_CHILD_CREDIT',	'AMOUNT_EDUCATION_CREDIT_PY2',	'AMOUNT_EXEMPTIONS_PY',	'AMOUNT_INCOME_TAX_WITHHELD_PY',	'AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION_PY2',	'AMOUNT_ORDINARY_DIVIDENDS_PY',	'AMOUNT_REFUND',	'AMOUNT_SALARIES_AND_WAGES_PY',	'AMOUNT_SCHE_PY',	'AMOUNT_SCHE_PY2',	'AMOUNT_SOCIAL_SEC_PY2',	'AMOUNT_TAX_DUE_PY',	'AMOUNT_TAXABLE_INCOME',	'AMOUNT_TAXABLE_SOCIAL_SEC_PY',	'AMOUNT_TOTAL_CREDITS_PY',	'AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES_PY',	'AUDIT_DEFENSE_FLAG',	'AUTH_NOT_COMPLETE_PY2',	'AUTH_NOT_COMPLETE_PY3',	'BUS_COGS_TOTAL_PY2',	'BUS_EXPENSE_EMPLOYEE_BENEFITS_PY2',	'BUS_GROSS_INCOME_PY3',	'BUS_NET_PROFIT',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'NON_CA_REFUND_TRANSFER_REVENUE',	'NUM_W2_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'SUP_PS_AT_RISK',	'SUP_PS_AT_RISK_PY2',	'SUP_PS_PARTNERSHIP_PY',	'SUP_PS_PARTNERSHIP_PY2',	'SUP_RE_EXPENSES_TOTAL',	'SUP_RE_EXPENSES_TOTAL_PY3',	'SUP_RE_INCOME_RENTS',	'SUP_RE_TYPE_ROYALTIES_PY3',	'SUP_RE_TYPE_SELF_RENTAL_PY',	'TTO_FLAG_PY2']
# bool_features=['ABANDONED',	'AUDIT_DEFENSE_FLAG',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'TTO_FLAG_PY2']
# catag_features=['TAX_DAY','TAX_DAY_PY','TAX_DAY_PY2','TAX_DAY_PY3','CHANNEL',	'CHANNEL_PY',	'COMPLETED_SKU',	'COMPLETED_SKU_PY',	'COMPLETED_SKU_PY2',	'COMPLETED_SKU_PY3',	'CUSTOMER_DEFINITION_ADJ',	'CUSTOMER_DEFINITION_ADJ_PY',	'CUSTOMER_DEFINITION_ADJ_PY2',	'ENTRY_PAGE_GROUP',	'ENTRY_PAGE_GROUP_PY',	'FED_FORM_TYPE',	'FED_FORM_TYPE_PY',	'FED_FORM_TYPE_PY2',	'FILING_STATUS',		'FILING_STATUS_PY',	'FILING_STATUS_PY2',	'FILING_STATUS_PY3',	'IMPORT_TYPE',	'IMPORT_TYPE_PY',	'IMPORT_TYPE_PY2',	'LAST_STATUS',	'LAST_STATUS_PY3',	'PRODUCT_EDITION_DESCRIPTION',	'PRODUCT_EDITION_DESCRIPTION_PY3',	'PRODUCT_ROLLUP',	'PRODUCT_ROLLUP_PY2',	'PRODUCT_ROLLUP_PY3',	'START_SKU_PY',	'START_SKU_PY2',	'START_SKU_ROLLUP_PY']
# cont_score_features=[	'AGE_DEPENDENT_MAX',	'AGE_DEPENDENT_MAX_PY',	'AGE_DEPENDENT_MAX_PY2',	'AGE_DEPENDENT_MIN_PY',	'AGE_DEPENDENT_MIN_PY2',	'AGE_SPOUSE_PY2',	'AGE_TAXPAYER_PY',	'AGE_TAXPAYER_PY2',	'AMOUNT_ADJUSTMENTS',	'AMOUNT_CHILD_CREDIT',	'AMOUNT_EDUCATION_CREDIT_PY2',	'AMOUNT_EXEMPTIONS_PY',	'AMOUNT_INCOME_TAX_WITHHELD_PY',	'AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION_PY2',	'AMOUNT_ORDINARY_DIVIDENDS_PY',	'AMOUNT_REFUND',	'AMOUNT_SALARIES_AND_WAGES_PY',	'AMOUNT_SCHE_PY',	'AMOUNT_SCHE_PY2',	'AMOUNT_SOCIAL_SEC_PY2',	'AMOUNT_TAX_DUE_PY',	'AMOUNT_TAXABLE_INCOME',	'AMOUNT_TAXABLE_SOCIAL_SEC_PY',	'AMOUNT_TOTAL_CREDITS_PY',	'AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES_PY',	'AUDIT_DEFENSE_FLAG',	'AUTH_NOT_COMPLETE_PY2',	'AUTH_NOT_COMPLETE_PY3',	'BUS_COGS_TOTAL_PY2',	'BUS_EXPENSE_EMPLOYEE_BENEFITS_PY2',	'BUS_GROSS_INCOME_PY3',	'BUS_NET_PROFIT',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'NON_CA_REFUND_TRANSFER_REVENUE',	'NUM_W2_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'SUP_PS_AT_RISK',	'SUP_PS_AT_RISK_PY2',	'SUP_PS_PARTNERSHIP_PY',	'SUP_PS_PARTNERSHIP_PY2',	'SUP_RE_EXPENSES_TOTAL',	'SUP_RE_EXPENSES_TOTAL_PY3',	'SUP_RE_INCOME_RENTS',	'SUP_RE_TYPE_ROYALTIES_PY3',	'SUP_RE_TYPE_SELF_RENTAL_PY',	'TTO_FLAG_PY2']
# bool_score_features=['AUDIT_DEFENSE_FLAG',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'TTO_FLAG_PY2']
# catag_score_features=['TAX_DAY','TAX_DAY_PY','TAX_DAY_PY2','TAX_DAY_PY3','CHANNEL',	'CHANNEL_PY',	'COMPLETED_SKU',	'COMPLETED_SKU_PY',	'COMPLETED_SKU_PY2',	'COMPLETED_SKU_PY3',	'CUSTOMER_DEFINITION_ADJ',	'CUSTOMER_DEFINITION_ADJ_PY',	'CUSTOMER_DEFINITION_ADJ_PY2',	'ENTRY_PAGE_GROUP',	'ENTRY_PAGE_GROUP_PY',	'FED_FORM_TYPE',	'FED_FORM_TYPE_PY',	'FED_FORM_TYPE_PY2',	'FILING_STATUS',	'FILING_STATUS_PY',	'FILING_STATUS_PY2',	'FILING_STATUS_PY3',	'IMPORT_TYPE',	'IMPORT_TYPE_PY',	'IMPORT_TYPE_PY2',	'LAST_STATUS',	'LAST_STATUS_PY3',	'PRODUCT_EDITION_DESCRIPTION',	'PRODUCT_EDITION_DESCRIPTION_PY3',	'PRODUCT_ROLLUP',	'PRODUCT_ROLLUP_PY2',	'PRODUCT_ROLLUP_PY3',	'START_SKU_PY',	'START_SKU_PY2',	'START_SKU_ROLLUP_PY']

# NEW
# cont_features=['ABANDONED','ACCEPTED_EFILE',	'AMOUNT_ADJUSTMENTS',	'AMOUNT_ALIMONY_INCOME',	'AMOUNT_ALIMONY_PAID',	'AMOUNT_CHARITABLE_CONTRIBUTIONS',	'AMOUNT_CHARITABLE_CONTRIBUTIONS_CASH',	'AMOUNT_CHARITABLE_CONTRIBUTIONS_NONCASH',	'AMOUNT_CHILD_CREDIT',	'AMOUNT_DEDUCTIBLE_SELF_EMPLOYMENT_TAX',	'AMOUNT_EARLY_WITHDRAWAL_PENALTY',	'AMOUNT_EITC',	'AMOUNT_EXEMPTIONS',	'AMOUNT_EXPENSES_DEDUCTION',	'AMOUNT_HOPE_CREDIT',	'AMOUNT_INCOME_TAX_WITHHELD',	'AMOUNT_IRA_DISTRIBUTIONS',	'AMOUNT_MEDICAL_DENTAL_EXPENSES',	'AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION',	'AMOUNT_MISC_DEDUCTIONS',	'AMOUNT_MORTGAGE_INTEREST_NON_1098',	'AMOUNT_MOVING_EXPENSE',	'AMOUNT_NT_COMBAT_PAY',	'AMOUNT_ORDINARY_DIVIDENDS',	'AMOUNT_OTHER_DEDUCTIBLE_EXPENSES',	'AMOUNT_PERSONAL_PROPERTY_TAXES',	'AMOUNT_QUALIFIED_DIVIDENDS',	'AMOUNT_SALARIES_AND_WAGES',	'AMOUNT_SCHE',	'AMOUNT_SELF_EMPLOYMENT_TAX',	'AMOUNT_STATE_LOCAL_TAX',	'AMOUNT_TAX_DUE',	'AMOUNT_TAXABLE_INCOME',	'AMOUNT_TAXABLE_INTEREST',	'AMOUNT_TAXABLE_IRA',	'AMOUNT_TAXABLE_SOCIAL_SEC',	'AMOUNT_TAXES_PAID',	'AMOUNT_TOTAL_CREDITS',	'AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES',	'AMOUNT_TOTAL_DEDUCTIONS',	'AMOUNT_TOTAL_INCOME',	'AMOUNT_TUITION',	'AMOUNT_UNEMPLOYMENT',	'AUDIT_DEFENSE_FLAG',	'AUDIT_DEFENSE_REVENUE',	'BUS_ACCOUNTING_METHOD_CASH',	'BUS_COGS_INVENTORY_END',	'BUS_COGS_INVENTORY_START',	'BUS_EXPENSE_ADVERTISING',	'BUS_EXPENSE_CAR',	'BUS_EXPENSE_CONTRACT_LABOR',	'BUS_EXPENSE_OTHER',	'BUS_EXPENSE_TOTAL',	'BUS_GROSS_INCOME',	'BUS_MATERIAL_PARTICIPATE',	'BUS_NET_PROFIT',	'BUS_REQUIRE_1099',	'BUS_VEHICLE_MILES_BUSINESS',	'BUS_VEHICLE_MILES_COMMUTE',	'BUS_VEHICLE_MILES_OTHER',	'BUS_VEHICLE_PERSONAL_ANOTHER',	'BUS_VEHICLE_PERSONAL_OFFDUTY',	'BUS_WILL_FILE_1099',	'CA_AUDIT_DEFENSE_FLAG',	'CA_REFUND_TRANSFER_FLAG',	'CA_REFUND_TRANSFER_REVENUE',	'FLAG_ITEMIZE_SEPARATELY',	'FLAG_OLD_OR_BLIND',	'FSCHA_FLAG',	'FSCHC_FLAG',	'FSCHCEZ_FLAG',	'FSCHD_FLAG',	'FSCHF_FLAG',	'MINDBENDER_FLAG',	'MINDBENDER_REVENUE',	'MISC1099_FLAG',	'NUM_DEPENDENTS',	'NUM_EXEMPTIONS',	'NUM_SCHE',	'NUM_W2',	'POST_TOTAL_VOTES',	'PRS_SCORE',	'REFUND_TRANSFER_FLAG',	'REJECT_COUNT',	'REQUIRED_TAKE_FLAG',	'RT_FLAG',	'STATE_ATTACH_COUNT',	'STATE_REVENUE',	'SUP_EST_TOTAL_INCOME',	'SUP_FARM_GROSS_INCOME',	'SUP_FARM_RENTAL_INCOME',	'SUP_PS_AT_RISK',	'SUP_PS_NONPASSIVE_INCOME',	'SUP_PS_NONPASSIVE_LOSS',	'SUP_PS_PARTNERSHIP',	'SUP_PS_TOTAL_INCOME',	'SUP_RE_EXPENSES_ADVERTISING',	'SUP_RE_EXPENSES_CLEANING',	'SUP_RE_EXPENSES_DEPRECIATION',	'SUP_RE_EXPENSES_TAXES',	'SUP_RE_EXPENSES_TOTAL',	'SUP_RE_EXPENSES_TRAVEL',	'SUP_RE_INCOME_RENTS',	'SUP_RE_REQUIRE_1099',	'SUP_RE_TOTAL_INCOME',	'SUP_RE_TYPE_LAND',	'SUP_RE_TYPE_MULTI_FAMILY',	'TOTAL_REVENUE'	]
# bool_features=['ABANDONED','ACCEPTED_EFILE',	'AUDIT_DEFENSE_FLAG',	'CA_AUDIT_DEFENSE_FLAG',	'CA_REFUND_TRANSFER_FLAG',	'FLAG_ITEMIZE_SEPARATELY',	'FLAG_OLD_OR_BLIND',	'FSCHA_FLAG',	'FSCHC_FLAG',	'FSCHCEZ_FLAG',	'FSCHD_FLAG',	'FSCHF_FLAG',	'MINDBENDER_FLAG',	'MISC1099_FLAG',	'REFUND_TRANSFER_FLAG',	'REQUIRED_TAKE_FLAG',	'RT_FLAG']
# catag_features=['CHANNEL',	'COMPLETED_SKU',	'ENTRY_PAGE_GROUP',	'FED_FORM_TYPE',	'FILING_STATUS',	'IMPORT_TYPE',	'LAST_STATUS',	'PRODUCT_EDITION_DESCRIPTION',	'PRODUCT_ROLLUP',	'START_SKU',	'TAX_DAY',	'TAX_WEEK']
# cont_score_features=['ACCEPTED_EFILE',	'AMOUNT_ADJUSTMENTS',	'AMOUNT_ALIMONY_INCOME',	'AMOUNT_ALIMONY_PAID',	'AMOUNT_CHARITABLE_CONTRIBUTIONS',	'AMOUNT_CHARITABLE_CONTRIBUTIONS_CASH',	'AMOUNT_CHARITABLE_CONTRIBUTIONS_NONCASH',	'AMOUNT_CHILD_CREDIT',	'AMOUNT_DEDUCTIBLE_SELF_EMPLOYMENT_TAX',	'AMOUNT_EARLY_WITHDRAWAL_PENALTY',	'AMOUNT_EITC',	'AMOUNT_EXEMPTIONS',	'AMOUNT_EXPENSES_DEDUCTION',	'AMOUNT_HOPE_CREDIT',	'AMOUNT_INCOME_TAX_WITHHELD',	'AMOUNT_IRA_DISTRIBUTIONS',	'AMOUNT_MEDICAL_DENTAL_EXPENSES',	'AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION',	'AMOUNT_MISC_DEDUCTIONS',	'AMOUNT_MORTGAGE_INTEREST_NON_1098',	'AMOUNT_MOVING_EXPENSE',	'AMOUNT_NT_COMBAT_PAY',	'AMOUNT_ORDINARY_DIVIDENDS',	'AMOUNT_OTHER_DEDUCTIBLE_EXPENSES',	'AMOUNT_PERSONAL_PROPERTY_TAXES',	'AMOUNT_QUALIFIED_DIVIDENDS',	'AMOUNT_SALARIES_AND_WAGES',	'AMOUNT_SCHE',	'AMOUNT_SELF_EMPLOYMENT_TAX',	'AMOUNT_STATE_LOCAL_TAX',	'AMOUNT_TAX_DUE',	'AMOUNT_TAXABLE_INCOME',	'AMOUNT_TAXABLE_INTEREST',	'AMOUNT_TAXABLE_IRA',	'AMOUNT_TAXABLE_SOCIAL_SEC',	'AMOUNT_TAXES_PAID',	'AMOUNT_TOTAL_CREDITS',	'AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES',	'AMOUNT_TOTAL_DEDUCTIONS',	'AMOUNT_TOTAL_INCOME',	'AMOUNT_TUITION',	'AMOUNT_UNEMPLOYMENT',	'AUDIT_DEFENSE_FLAG',	'AUDIT_DEFENSE_REVENUE',	'BUS_ACCOUNTING_METHOD_CASH',	'BUS_COGS_INVENTORY_END',	'BUS_COGS_INVENTORY_START',	'BUS_EXPENSE_ADVERTISING',	'BUS_EXPENSE_CAR',	'BUS_EXPENSE_CONTRACT_LABOR',	'BUS_EXPENSE_OTHER',	'BUS_EXPENSE_TOTAL',	'BUS_GROSS_INCOME',	'BUS_MATERIAL_PARTICIPATE',	'BUS_NET_PROFIT',	'BUS_REQUIRE_1099',	'BUS_VEHICLE_MILES_BUSINESS',	'BUS_VEHICLE_MILES_COMMUTE',	'BUS_VEHICLE_MILES_OTHER',	'BUS_VEHICLE_PERSONAL_ANOTHER',	'BUS_VEHICLE_PERSONAL_OFFDUTY',	'BUS_WILL_FILE_1099',	'CA_AUDIT_DEFENSE_FLAG',	'CA_REFUND_TRANSFER_FLAG',	'CA_REFUND_TRANSFER_REVENUE',	'FLAG_ITEMIZE_SEPARATELY',	'FLAG_OLD_OR_BLIND',	'FSCHA_FLAG',	'FSCHC_FLAG',	'FSCHCEZ_FLAG',	'FSCHD_FLAG',	'FSCHF_FLAG',	'MINDBENDER_FLAG',	'MINDBENDER_REVENUE',	'MISC1099_FLAG',	'NUM_DEPENDENTS',	'NUM_EXEMPTIONS',	'NUM_SCHE',	'NUM_W2',	'POST_TOTAL_VOTES',	'PRS_SCORE',	'REFUND_TRANSFER_FLAG',	'REJECT_COUNT',	'REQUIRED_TAKE_FLAG',	'RT_FLAG',	'STATE_ATTACH_COUNT',	'STATE_REVENUE',	'SUP_EST_TOTAL_INCOME',	'SUP_FARM_GROSS_INCOME',	'SUP_FARM_RENTAL_INCOME',	'SUP_PS_AT_RISK',	'SUP_PS_NONPASSIVE_INCOME',	'SUP_PS_NONPASSIVE_LOSS',	'SUP_PS_PARTNERSHIP',	'SUP_PS_TOTAL_INCOME',	'SUP_RE_EXPENSES_ADVERTISING',	'SUP_RE_EXPENSES_CLEANING',	'SUP_RE_EXPENSES_DEPRECIATION',	'SUP_RE_EXPENSES_TAXES',	'SUP_RE_EXPENSES_TOTAL',	'SUP_RE_EXPENSES_TRAVEL',	'SUP_RE_INCOME_RENTS',	'SUP_RE_REQUIRE_1099',	'SUP_RE_TOTAL_INCOME',	'SUP_RE_TYPE_LAND',	'SUP_RE_TYPE_MULTI_FAMILY',	'TOTAL_REVENUE']
# bool_score_features=['ACCEPTED_EFILE',	'AUDIT_DEFENSE_FLAG',	'CA_AUDIT_DEFENSE_FLAG',	'CA_REFUND_TRANSFER_FLAG',	'FLAG_ITEMIZE_SEPARATELY',	'FLAG_OLD_OR_BLIND',	'FSCHA_FLAG',	'FSCHC_FLAG',	'FSCHCEZ_FLAG',	'FSCHD_FLAG',	'FSCHF_FLAG',	'MINDBENDER_FLAG',	'MISC1099_FLAG',	'REFUND_TRANSFER_FLAG',	'REQUIRED_TAKE_FLAG',	'RT_FLAG']
# catag_score_features=['CHANNEL',	'COMPLETED_SKU',	'ENTRY_PAGE_GROUP',	'FED_FORM_TYPE',	'FILING_STATUS',	'IMPORT_TYPE',	'LAST_STATUS',	'PRODUCT_EDITION_DESCRIPTION',	'PRODUCT_ROLLUP',	'START_SKU',	'TAX_DAY',	'TAX_WEEK']

# # NEW V2
# cont_features=['ABANDONED','AUDIT_DEFENSE_FLAG',	'FLAG_OLD_OR_BLIND',	'FSCHA_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'REQUIRED_TAKE_FLAG',	'student_taxpayer',	'AGE_DEPENDENT_MIN',	'AMOUNT_EITC',	'AMOUNT_ESTIMATED_TAX_PENALTY',	'AMOUNT_EXEMPTIONS',	'AMOUNT_EXPENSES_DEDUCTION',	'AMOUNT_INCOME_TAX_WITHHELD',	'AMOUNT_TAXABLE_INCOME',	'AMOUNT_TOTAL_DEDUCTIONS',	'AMOUNT_TOTAL_TAX',	'BUS_COGS_TOTAL',	'REJECT_COUNT',	'SUP_PS_PARTNERSHIP',	'ACCEPTED_EFILE',	'AGE_DEPENDENT_MAX',	'AGE_TAXPAYER',	'AMOUNT_ADJUSTMENTS',	'AMOUNT_BUSINESS_INCOME',	'AMOUNT_CHARITABLE_CONTRIBUTIONS',	'AMOUNT_CHARITABLE_CONTRIBUTIONS_CASH',	'AMOUNT_EDUCATION_CREDIT',	'AMOUNT_EXCESS_SS_RRTA_WITHHELD',	'AMOUNT_MEDICAL_DENTAL_EXPENSES',	'AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION',	'AMOUNT_ORDINARY_DIVIDENDS',	'AMOUNT_OTHER_TAXES',	'AMOUNT_QUALIFIED_DIVIDENDS',	'BUS_COGS_INVENTORY_END',	'BUS_COGS_INVENTORY_START',	'BUS_COGS_PURCHASES',	'BUS_EXPENSE_EMPLOYEE_BENEFITS',	'BUS_EXPENSE_PENSION',	'BUS_GROSS_INCOME',	'BUS_REQUIRE_1099',	'BUS_VEHICLE_MILES_BUSINESS',	'BUS_WILL_FILE_1099',	'CA_REFUND_TRANSFER_FLAG',	'cook_taxpayer',	'customerservice_taxpayer',	'FSCHC_FLAG',	'FSCHCEZ_FLAG',	'FSCHD_FLAG',	'FSCHE_FLAG',	'homemaker_spouse',	'laborer_taxpayer',	'manager_taxpayer',	'NON_CA_AUDIT_DEFENSE_FLAG',	'sales_taxpayer',	'server_taxpayer',	'SUP_EST_TOTAL_INCOME',	'SUP_FARM_GROSS_INCOME',	'SUP_FARM_RENTAL_INCOME',	'SUP_PS_AT_RISK',	'SUP_RE_EXPENSES_TOTAL',	'SUP_RE_TOTAL_INCOME',	'SUP_RE_TYPE_ROYALTIES',	'teacher_taxpayer'	]
# bool_features=['ABANDONED','AUDIT_DEFENSE_FLAG',	'FLAG_OLD_OR_BLIND',	'FSCHA_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'REQUIRED_TAKE_FLAG',	'student_taxpayer',	'CA_REFUND_TRANSFER_FLAG',	'cook_taxpayer',	'customerservice_taxpayer',	'FSCHC_FLAG',	'FSCHCEZ_FLAG',	'FSCHD_FLAG',	'FSCHE_FLAG',	'homemaker_spouse',	'laborer_taxpayer',	'manager_taxpayer',	'NON_CA_AUDIT_DEFENSE_FLAG',	'sales_taxpayer',	'server_taxpayer',	'teacher_taxpayer']
# catag_features=['CHANNEL',	'ENTRY_PAGE_GROUP',	'START_SKU',	'IMPORT_TYPE',	'FED_FORM_TYPE',	'PRODUCT_EDITION_DESCRIPTION',	'PRODUCT_ROLLUP',	'COMPLETED_SKU',	'LAST_STATUS',	'DMA_AREA',	'FILING_STATUS',	'NUM_EXEMPTIONS',	'NUM_W2',	'TAX_DAY']
# cont_score_features=['AUDIT_DEFENSE_FLAG',	'FLAG_OLD_OR_BLIND',	'FSCHA_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'REQUIRED_TAKE_FLAG',	'student_taxpayer',	'AGE_DEPENDENT_MIN',	'AMOUNT_EITC',	'AMOUNT_ESTIMATED_TAX_PENALTY',	'AMOUNT_EXEMPTIONS',	'AMOUNT_EXPENSES_DEDUCTION',	'AMOUNT_INCOME_TAX_WITHHELD',	'AMOUNT_TAXABLE_INCOME',	'AMOUNT_TOTAL_DEDUCTIONS',	'AMOUNT_TOTAL_TAX',	'BUS_COGS_TOTAL',	'REJECT_COUNT',	'SUP_PS_PARTNERSHIP',	'ACCEPTED_EFILE',	'AGE_DEPENDENT_MAX',	'AGE_TAXPAYER',	'AMOUNT_ADJUSTMENTS',	'AMOUNT_BUSINESS_INCOME',	'AMOUNT_CHARITABLE_CONTRIBUTIONS',	'AMOUNT_CHARITABLE_CONTRIBUTIONS_CASH',	'AMOUNT_EDUCATION_CREDIT',	'AMOUNT_EXCESS_SS_RRTA_WITHHELD',	'AMOUNT_MEDICAL_DENTAL_EXPENSES',	'AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION',	'AMOUNT_ORDINARY_DIVIDENDS',	'AMOUNT_OTHER_TAXES',	'AMOUNT_QUALIFIED_DIVIDENDS',	'BUS_COGS_INVENTORY_END',	'BUS_COGS_INVENTORY_START',	'BUS_COGS_PURCHASES',	'BUS_EXPENSE_EMPLOYEE_BENEFITS',	'BUS_EXPENSE_PENSION',	'BUS_GROSS_INCOME',	'BUS_REQUIRE_1099',	'BUS_VEHICLE_MILES_BUSINESS',	'BUS_WILL_FILE_1099',	'CA_REFUND_TRANSFER_FLAG',	'cook_taxpayer',	'customerservice_taxpayer',	'FSCHC_FLAG',	'FSCHCEZ_FLAG',	'FSCHD_FLAG',	'FSCHE_FLAG',	'homemaker_spouse',	'laborer_taxpayer',	'manager_taxpayer',	'NON_CA_AUDIT_DEFENSE_FLAG',	'sales_taxpayer',	'server_taxpayer',	'SUP_EST_TOTAL_INCOME',	'SUP_FARM_GROSS_INCOME',	'SUP_FARM_RENTAL_INCOME',	'SUP_PS_AT_RISK',	'SUP_RE_EXPENSES_TOTAL',	'SUP_RE_TOTAL_INCOME',	'SUP_RE_TYPE_ROYALTIES',	'teacher_taxpayer']
# bool_score_features=['AUDIT_DEFENSE_FLAG',	'FLAG_OLD_OR_BLIND',	'FSCHA_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'REQUIRED_TAKE_FLAG',	'student_taxpayer',	'CA_REFUND_TRANSFER_FLAG',	'cook_taxpayer',	'customerservice_taxpayer',	'FSCHC_FLAG',	'FSCHCEZ_FLAG',	'FSCHD_FLAG',	'FSCHE_FLAG',	'homemaker_spouse',	'laborer_taxpayer',	'manager_taxpayer',	'NON_CA_AUDIT_DEFENSE_FLAG',	'sales_taxpayer',	'server_taxpayer',	'teacher_taxpayer']
# catag_score_features=['CHANNEL',	'ENTRY_PAGE_GROUP',	'START_SKU',	'IMPORT_TYPE',	'FED_FORM_TYPE',	'PRODUCT_EDITION_DESCRIPTION',	'PRODUCT_ROLLUP',	'COMPLETED_SKU',	'LAST_STATUS',	'DMA_AREA',	'FILING_STATUS',	'NUM_EXEMPTIONS',	'NUM_W2',	'TAX_DAY']

# cont_features=[	'ABANDONED',	'AGE_DEPENDENT_MAX',	'AGE_DEPENDENT_MAX_PY',	'AGE_DEPENDENT_MAX_PY2',	'AGE_DEPENDENT_MIN_PY',	'AGE_DEPENDENT_MIN_PY2',	'AGE_SPOUSE_PY2',	'AGE_TAXPAYER_PY',	'AGE_TAXPAYER_PY2',	'AMOUNT_ADJUSTMENTS',	'AMOUNT_CHILD_CREDIT',	'AMOUNT_EDUCATION_CREDIT_PY2',	'AMOUNT_EXEMPTIONS_PY',	'AMOUNT_INCOME_TAX_WITHHELD_PY',	'AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION_PY2',	'AMOUNT_ORDINARY_DIVIDENDS_PY',	'AMOUNT_REFUND',	'AMOUNT_SALARIES_AND_WAGES_PY',	'AMOUNT_SCHE_PY',	'AMOUNT_SCHE_PY2',	'AMOUNT_SOCIAL_SEC_PY2',	'AMOUNT_TAX_DUE_PY',	'AMOUNT_TAXABLE_INCOME',	'AMOUNT_TAXABLE_SOCIAL_SEC_PY',	'AMOUNT_TOTAL_CREDITS_PY',	'AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES_PY',	'AUDIT_DEFENSE_FLAG',	'AUTH_NOT_COMPLETE_PY2',	'AUTH_NOT_COMPLETE_PY3',	'BUS_COGS_TOTAL_PY2',	'BUS_EXPENSE_EMPLOYEE_BENEFITS_PY2',	'BUS_GROSS_INCOME_PY3',	'BUS_NET_PROFIT',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'NON_CA_REFUND_TRANSFER_REVENUE',	'NUM_W2_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'SUP_PS_AT_RISK',	'SUP_PS_AT_RISK_PY2',	'SUP_PS_PARTNERSHIP_PY',	'SUP_PS_PARTNERSHIP_PY2',	'SUP_RE_EXPENSES_TOTAL',	'SUP_RE_EXPENSES_TOTAL_PY3',	'SUP_RE_INCOME_RENTS',	'SUP_RE_TYPE_ROYALTIES_PY3',	'SUP_RE_TYPE_SELF_RENTAL_PY',	'TTO_FLAG_PY2']
# bool_features=['ABANDONED',	'AUDIT_DEFENSE_FLAG',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'TTO_FLAG_PY2']
# catag_features=['TAX_DAY','TAX_DAY_PY','TAX_DAY_PY2','TAX_DAY_PY3','CHANNEL',	'CHANNEL_PY',	'COMPLETED_SKU',	'COMPLETED_SKU_PY',	'COMPLETED_SKU_PY2',	'COMPLETED_SKU_PY3',	'CUSTOMER_DEFINITION_ADJ',	'CUSTOMER_DEFINITION_ADJ_PY',	'CUSTOMER_DEFINITION_ADJ_PY2',	'ENTRY_PAGE_GROUP',	'ENTRY_PAGE_GROUP_PY',	'FED_FORM_TYPE',	'FED_FORM_TYPE_PY',	'FED_FORM_TYPE_PY2',	'FILING_STATUS',		'FILING_STATUS_PY',	'FILING_STATUS_PY2',	'FILING_STATUS_PY3',	'IMPORT_TYPE',	'IMPORT_TYPE_PY',	'IMPORT_TYPE_PY2',	'LAST_STATUS',	'LAST_STATUS_PY3',	'PRODUCT_EDITION_DESCRIPTION',	'PRODUCT_EDITION_DESCRIPTION_PY3',	'PRODUCT_ROLLUP',	'PRODUCT_ROLLUP_PY2',	'PRODUCT_ROLLUP_PY3',	'START_SKU_PY',	'START_SKU_PY2',	'START_SKU_ROLLUP_PY']
# cont_score_features=[	'AGE_DEPENDENT_MAX',	'AGE_DEPENDENT_MAX_PY',	'AGE_DEPENDENT_MAX_PY2',	'AGE_DEPENDENT_MIN_PY',	'AGE_DEPENDENT_MIN_PY2',	'AGE_SPOUSE_PY2',	'AGE_TAXPAYER_PY',	'AGE_TAXPAYER_PY2',	'AMOUNT_ADJUSTMENTS',	'AMOUNT_CHILD_CREDIT',	'AMOUNT_EDUCATION_CREDIT_PY2',	'AMOUNT_EXEMPTIONS_PY',	'AMOUNT_INCOME_TAX_WITHHELD_PY',	'AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION_PY2',	'AMOUNT_ORDINARY_DIVIDENDS_PY',	'AMOUNT_REFUND',	'AMOUNT_SALARIES_AND_WAGES_PY',	'AMOUNT_SCHE_PY',	'AMOUNT_SCHE_PY2',	'AMOUNT_SOCIAL_SEC_PY2',	'AMOUNT_TAX_DUE_PY',	'AMOUNT_TAXABLE_INCOME',	'AMOUNT_TAXABLE_SOCIAL_SEC_PY',	'AMOUNT_TOTAL_CREDITS_PY',	'AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES_PY',	'AUDIT_DEFENSE_FLAG',	'AUTH_NOT_COMPLETE_PY2',	'AUTH_NOT_COMPLETE_PY3',	'BUS_COGS_TOTAL_PY2',	'BUS_EXPENSE_EMPLOYEE_BENEFITS_PY2',	'BUS_GROSS_INCOME_PY3',	'BUS_NET_PROFIT',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'NON_CA_REFUND_TRANSFER_REVENUE',	'NUM_W2_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'SUP_PS_AT_RISK',	'SUP_PS_AT_RISK_PY2',	'SUP_PS_PARTNERSHIP_PY',	'SUP_PS_PARTNERSHIP_PY2',	'SUP_RE_EXPENSES_TOTAL',	'SUP_RE_EXPENSES_TOTAL_PY3',	'SUP_RE_INCOME_RENTS',	'SUP_RE_TYPE_ROYALTIES_PY3',	'SUP_RE_TYPE_SELF_RENTAL_PY',	'TTO_FLAG_PY2']
# bool_score_features=['AUDIT_DEFENSE_FLAG',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'TTO_FLAG_PY2']
# catag_score_features=['TAX_DAY','TAX_DAY_PY','TAX_DAY_PY2','TAX_DAY_PY3','CHANNEL',	'CHANNEL_PY',	'COMPLETED_SKU',	'COMPLETED_SKU_PY',	'COMPLETED_SKU_PY2',	'COMPLETED_SKU_PY3',	'CUSTOMER_DEFINITION_ADJ',	'CUSTOMER_DEFINITION_ADJ_PY',	'CUSTOMER_DEFINITION_ADJ_PY2',	'ENTRY_PAGE_GROUP',	'ENTRY_PAGE_GROUP_PY',	'FED_FORM_TYPE',	'FED_FORM_TYPE_PY',	'FED_FORM_TYPE_PY2',	'FILING_STATUS',	'FILING_STATUS_PY',	'FILING_STATUS_PY2',	'FILING_STATUS_PY3',	'IMPORT_TYPE',	'IMPORT_TYPE_PY',	'IMPORT_TYPE_PY2',	'LAST_STATUS',	'LAST_STATUS_PY3',	'PRODUCT_EDITION_DESCRIPTION',	'PRODUCT_EDITION_DESCRIPTION_PY3',	'PRODUCT_ROLLUP',	'PRODUCT_ROLLUP_PY2',	'PRODUCT_ROLLUP_PY3',	'START_SKU_PY',	'START_SKU_PY2',	'START_SKU_ROLLUP_PY']


for x in range(12,120,1):
    scoring_data = "SELECT  ABANDONED,COST_PER_CUST,	START_DEVICE_TYPE,	BUS_WILL_FILE_1099,FILING_STATUS,	CUSTOMER_TYPE,	AGE_SPOUSE,	PRS_SCORE,	AGE_TAXPAYER,	AMOUNT_STUDENT_LOAN_INTEREST_DEDUCTION,	AMOUNT_UNEMPLOYMENT,	COMPLETED_SKU,	DMA_AREA,	FED_FORM_TYPE,	FIRST_COMPLETE_APP_TYPE,	FIRST_COMPLETE_DEVICE_TYPE,	FLAG_ITEMIZED_DEDUCTIONS,	IMPORT_TYPE,	LAST_STATUS,	NEAUTH_DEVICE_TYPE,	NUM_DEPENDENTS,	NUM_EXEMPTIONS,	NUM_SCHC,	NUM_SCHE,	NUM_W2,	PRODUCT_EDITION_DESCRIPTION,	REJECT_COUNT,	SESSIONS_TO_COMPLETE,	START_SKU,	STATE_ATTACH_COUNT,	TOTAL_REVENUE,	VAUTH_DEVICE_TYPE FROM  CTG_ANALYTICS_WS.SM_RETENTION_MODEL  WHERE   TAX_YEAR=2014  AND CUSTOMER_DEFINITION_ADJ='NEW TO TURBOTAX' order by random() limit 1000"
    data = "SELECT  ABANDONED,COST_PER_CUST,	START_DEVICE_TYPE,	BUS_WILL_FILE_1099,FILING_STATUS,	CUSTOMER_TYPE,	AGE_SPOUSE,	PRS_SCORE,	AGE_TAXPAYER,	AMOUNT_STUDENT_LOAN_INTEREST_DEDUCTION,	AMOUNT_UNEMPLOYMENT,	COMPLETED_SKU,	DMA_AREA,	FED_FORM_TYPE,	FIRST_COMPLETE_APP_TYPE,	FIRST_COMPLETE_DEVICE_TYPE,	FLAG_ITEMIZED_DEDUCTIONS,	IMPORT_TYPE,	LAST_STATUS,	NEAUTH_DEVICE_TYPE,	NUM_DEPENDENTS,	NUM_EXEMPTIONS,	NUM_SCHC,	NUM_SCHE,	NUM_W2,	PRODUCT_EDITION_DESCRIPTION,	REJECT_COUNT,	SESSIONS_TO_COMPLETE,	START_SKU,	STATE_ATTACH_COUNT,	TOTAL_REVENUE,	VAUTH_DEVICE_TYPE FROM  CTG_ANALYTICS_WS.SM_RETENTION_MODEL  WHERE   TAX_YEAR=2015 AND CUSTOMER_DEFINITION_ADJ='NEW TO TURBOTAX' order by random() limit 1000"


    def import_scoring_data(scoring_data, cont_score_features, bool_score_features,catag_score_features):
        df = pd.read_sql(scoring_data, conn, index_col='CUSTOMER_KEY',coerce_float=False)
        # df = df_raw[df_raw['TAX_YEAR'] == 2016]
        df_cont=df[cont_score_features]
        # print df_cont

        df_cont.columns = df_cont.columns.str.strip()
        df_cont.fillna(value=0,inplace=True)
        df_cont.replace(to_replace=('(null)', 'NA', 'None'), value=0)
        df_cont = df_cont.astype(float)


        df_bool = df_cont[bool_score_features]
        df_cont.drop(bool_score_features, axis=1, inplace=True)

        index_df=pd.DataFrame(df_cont.reset_index(level=['CUSTOMER_KEY']), columns=['CUSTOMER_KEY'])

        data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
        data_scaled=pd.concat([data_scaled,index_df], axis=1)
        data_scaled.set_index('CUSTOMER_KEY', inplace=True)


        df_char = df[catag_score_features]
        df_char=df_char.astype(object)


        df_char.columns = df_char.columns.str.strip()
        df_char.fillna(value='-1', inplace=True)
        df_char.replace(to_replace=('(null)', 'NA', 'None','',' ','\t'), value='-1')

        # print list(df_char)
        # print df_char
        #
        just_dummies = pd.get_dummies(df_char)
        # print 'SCORING...', just_dummies
        df_trans = pd.concat([df_bool, just_dummies,  data_scaled], axis=1)




        return df_trans, df_cont


                                                # ***************************TEST************************************************
                                                # *******************************************************************************
                                                # *******************************************************************************


    def import_data(data, cont_features, bool_features, catag_features,  scoring_df):
        # query = "SELECT * FROM CTG_ANALYTICS_WS.SM_TXML_TY13_TY14_S where  CUSTOMER_DEFINITION_ADJ IN ('NEW TO TURBOTAX')  ORDER BY RANDOM() LIMIT 5000"
        df = pd.read_sql(data, conn, index_col='CUSTOMER_KEY',coerce_float=False)
        # df = df_raw[df_raw['TAX_YEAR'] != 2016]

        df_cont=df[cont_features]
        df_cont.columns = df_cont.columns.str.strip()
        # print df_cont
        df_cont.fillna(value=0,inplace=True)
        df_cont.replace(to_replace=('(null)', 'NA'), value=0)

        df_cont=df_cont.astype(float)

        df_bool=df_cont[bool_features]
        df_cont.drop(bool_features, axis=1, inplace=True)

        # ch2 = SelectKBest(chi2, k=10)
        # df_bool=df_bool.astype(object)
        # print df_bool.dtypes
        # b = df_bool['ABANDONED']
        # df_bool.drop(['ABANDONED'], axis=1, inplace=True)
        # a=df_bool
        # x = ch2.fit(a,b)
        # df_bool.reset_index(['CUSTOMER_KEY'], inplace=True)
        # df_bool=pd.concat([df_bool['ABANDONED'], x], axis=1)
        # df_bool.set_index('CUSTOMER_KEY', inplace=True)
        # print df_bool

        index_df = pd.DataFrame(df_cont.reset_index(level=['CUSTOMER_KEY']), columns=['CUSTOMER_KEY'])
        print 'df_cont done'
        data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
        data_scaled = pd.concat([data_scaled, index_df], axis=1)
        data_scaled.set_index('CUSTOMER_KEY', inplace=True)
        print 'data_scaled done'

        df_char=df[catag_features]
        df_char.columns = df_char.columns.str.strip()
        df_char.fillna(value='-1',inplace=True)
        df_char.replace(to_replace=('(null)', 'NA'), value='-1')
        just_dummies=pd.get_dummies(df_char)
        print 'just_dummies done'
        df_trans=pd.concat([df_bool,just_dummies, data_scaled],axis=1)
        print 'df_trans done'
        new_list=list(set(list(df_trans)) & set(list(scoring_df)))
        print 'new feature list done'
        df_trans_pca2=df_trans[new_list]
        print list(df_trans_pca2)
        return df_trans_pca2, df_bool, df_cont

    def scoring_data_intersection(df_no_pca, scoring_df):

        new_list2=list(set(list(df_no_pca)) & set(list(scoring_df)))
        df_scoring2=scoring_df[new_list2]
        print "intersection of dataframes is done"

        return df_scoring2

    def pca_code(df_no_pca,df_cont, do_pca):

        if do_pca=='True':
            pca = PCA(n_components=100, random_state=np.random.RandomState(0))
            pca.fit(df_no_pca)
            x3 = pca.transform(df_no_pca)
            string = "pca_"
            pca_column_name=[string+`i` for i in range(x3.shape[1])]
            df1=pd.DataFrame(x3, columns=pca_column_name)
            df_no_pca.reset_index(['CUSTOMER_KEY'], inplace=True)
            df_no_pca.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
            reduced_df = pd.DataFrame(pca.components_, columns=df_no_pca.columns, index=pca_column_name)
            sig_features = list(set(reduced_df.idxmax(axis=1).values))
            df_final = df_no_pca[sig_features]
            bool=pd.DataFrame(bool_df['ABANDONED'], columns=['ABANDONED'])
            bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)
            df = pd.concat([df_final, bool['ABANDONED']], axis=1)

        else:
            df=pd.concat([df_no_pca, bool_df['ABANDONED']], axis=1)
            pca=()
        return df, pca, df_cont


    def bin_pca(df_pca, bool_df, df_cont, j):
        b_pca=j
        if b_pca=='True':
            pca_leng = {}

            # df_pca.drop(['ABANDONED'], inplace=True,axis=1, errors='ignore')
            print df_cont.dtypes
            lists=list(df_cont)
            print lists

            for k in lists:

                bool = pd.DataFrame(bool_df['ABANDONED'], columns=['ABANDONED'])
                # print bool
                # bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)

                df_bin = pd.concat([df_cont[k], bool['ABANDONED']], axis=1)


                dict = bin(df_bin)

                leng = len(dict)


                pca_level = 'pcl_'
                labels = [pca_level + `r` for r in range(leng)]
                print k, leng, labels
                print df_cont[k]
                df_pca[k]= pd.cut(df_cont[k], bins=leng, labels=labels)
                print df_cont[k]
                pca_leng[k] = leng

            df_trans_pca_dummy = pd.get_dummies(df_cont)
            bool = pd.DataFrame(bool_df['ABANDONED'], columns=['ABANDONED'])
            bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)
            df_pca.drop(lists, axis=1, inplace=True)
            training_df = pd.concat([df_trans_pca_dummy, bool['ABANDONED'],df_pca], axis=1)


        else:
            df_no_pca=df_pca
            training_df=pd.concat([df_no_pca, bool_df['ABANDONED']], axis=1)
            pca_leng={}

        return   training_df, pca_leng


    def get_arrays(dummy_pca,df_pca,df_no_pca, bool_df, do_pca, b_pca):

        response_feature = ['ABANDONED']
        if do_pca=='True' and b_pca=='True':
            Y = dummy_pca['ABANDONED']
            X = dummy_pca.drop(response_feature, 1)
            y = Y
            x = X

        elif do_pca=='True' and b_pca!='True':

            Y = df_pca['ABANDONED']
            X = df_pca.drop(response_feature, 1)
            y = Y
            x = X

        else:
            Y = bool_df['ABANDONED']
            X = df_no_pca
            y = Y
            x = X

        return x, y


    def feature_clustering(x,y,fc):
        if fc=='True':
            plsca = PLSRegression(n_components=200)
            plsca.fit(x,y)
            x3 = plsca.transform(x)
            string = "pls_"
            pls_column_name = [string + `i` for i in range(x3.shape[1])]
            df1 = pd.DataFrame(x3, columns=pls_column_name)
            plsca_df = pd.DataFrame(plsca.x_weights_)
            plsca_trans = plsca_df.transpose()
            x.reset_index(['CUSTOMER_KEY'], inplace=True)
            x.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
            reduced_df = pd.DataFrame(plsca_trans.values, columns=x.columns, index=pls_column_name)
            sig_features = list(set(reduced_df.idxmax(axis=1).values))

            df_final = x[sig_features]
            bool = pd.DataFrame(bool_df['ABANDONED'], columns=['ABANDONED'])
            bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)
            df = pd.concat([df_final, bool['ABANDONED']], axis=1)


            # plsca = PLSRegression(n_components=100)
            # plsca.fit(data_scaled, Y)
            # x_pls = plsca.transform(data_scaled)
            # string = "pls_"
            # x_pls_column_name = [string + `i` for i in range(x_pls.shape[1])]
            # plsca_df = pd.DataFrame(plsca.x_weights_)
            # plsca_trans = plsca_df.transpose()
            # x_pls_reduced_df = pd.DataFrame(plsca_trans.values, columns=X.columns, index=x_pls_column_name)
            # pls_sig_features = list(set(x_pls_reduced_df.idxmax(axis=1).values))
            # pls_final = clust_df[pls_sig_features]
            # pls_df = reduced_df[pls_sig_features]

        else:
            df_final=x
            bool = pd.DataFrame(bool_df['ABANDONED'], columns=['ABANDONED'])
            plsca=[]


        return df_final,bool['ABANDONED'],plsca

    def algorithm(x,y):

        names = [
            # "Nearest Neighbors" ,
            # "Decision_Tree",
            "Random_Forest"
            # "NeuralNetwork"
            # "AdaBoost",
            #  "Naive Bayes",
            # "Bernouli Niave Bayes",
            # "QDA",
            # "Bagging" ,
            # "ERT",
            # "GB"
        ]

        classifiers = [
            # KNeighborsClassifier(n_neighbors=20, leaf_size=1),
            # DecisionTreeClassifier(criterion='entropy'),
            RandomForestClassifier(criterion='entropy', n_estimators=200)
            # MLPClassifier(alpha=1e-5, random_state = random_state)
            # AdaBoostClassifier(n_estimators=100),
            # GaussianNB(),
            # BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True),
            # QuadraticDiscriminantAnalysis()
            # BaggingClassifier(bootstrap_features=True,random_state=np.random.RandomState(0)),
            # ExtraTreesClassifier(criterion='entropy', random_state=np.random.RandomState(0)),
            # GradientBoostingClassifier(n_estimators=1000, max_depth=10000, random_state= np.random.RandomState(0))
        ]

        models=[]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.4, random_state=np.random.RandomState(0))
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            precision = average_precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc=roc_auc_score(y_test, y_pred)
            f1=f1_score(y_test, y_pred)
            tn, fp, fn, tp=confusion_matrix(y_test, y_pred).ravel()
            f1 = f1_score(y_test, y_pred)
            # f1_avg=np.mean(cross_val_score(clf, X_train, y_train, scoring="f1", cv=3, n_jobs=1))
            # auc = np.mean(cross_val_score(clf, X_train, y_train, scoring="roc_auc", cv=3, n_jobs=1))
            # precision_avg = np.mean(cross_val_score(clf, X_train, y_train, scoring="precision", cv=3, n_jobs=1))
            # recall_avg = np.mean(cross_val_score(clf, X_train, y_train, scoring="recall", cv=3, n_jobs=1))
            print name , precision, recall, f1, auc, tn, fp, fn, tp
            naming = list(X_train)
            feature_df = pd.DataFrame(clf.feature_importances_, columns=['sig'], index=naming).sort_values(['sig'],ascending=False)
            print feature_df
                # , "recall_avg ", recall_avg, "precision_avg ", precision_avg

            models.append(clf)
            # return name, ' model', ' precision score', precision, ' recall score', recall, ' f1 ', f1
        return models, names



    def transform_to_pca(data, fitted_pca, i):
        do_pca=i
        if do_pca == 'True':
            scoring_data=fitted_pca.transform(data)
            string = "pca_"
            pca_column_name = [string + `i` for i in range(scoring_data.shape[1])]
            df = pd.DataFrame(scoring_data, columns=pca_column_name)
            data.reset_index(['CUSTOMER_KEY'], inplace=True)

            data.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
            reduced_df = pd.DataFrame(fitted_pca.components_, columns=data.columns, index=pca_column_name)

            sig_features = list(set(reduced_df.idxmax(axis=1).values))
            df = data[sig_features]

        else:
            df=data

        return df

    def bin_pca_score_set(df_trans_pca, length_dict, j):
        b_pca=j
        if b_pca=='True':
            pca_leng={}
            leng = len(length_dict)
            for i in df_trans_pca.columns:
                if i in length_dict:
                    pca_level = 'pcl_'
                    labels = [pca_level + `r` for r in range(length_dict[i])]
                    df_trans_pca[i] = pd.cut(df_trans_pca[i], bins=length_dict[i], labels=labels)
                    pca_leng[i] = leng
                elif i not in length_dict:
                    df_trans_pca.drop([i], axis=1, inplace=True)
            df_trans_pca_dummy = pd.get_dummies(df_trans_pca)
            scoring_df_trans = df_trans_pca_dummy
            print scoring_df_trans

        else:
            scoring_df_trans=df_trans_pca

        return scoring_df_trans

    def get_scoring_arrays(scoring_df_trans):
        x_score = scoring_df_trans
        return x_score

    def score_set_feature_selection(x, plsca, k):
        fc=k
        if fc=='True':
            x3 = plsca.transform(x)
            string = "pls_"
            pls_column_name = [string + `i` for i in range(x3.shape[1])]
            df1 = pd.DataFrame(x3, columns=pls_column_name)
            plsca_df = pd.DataFrame(plsca.x_weights_)
            plsca_trans = plsca_df.transpose()

            x.reset_index(['CUSTOMER_KEY'], inplace=True)
            index = x['CUSTOMER_KEY']
            x.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
            reduced_df = pd.DataFrame(plsca_trans.values, columns=x.columns, index=pls_column_name)
            sig_features = list(set(reduced_df.idxmax(axis=1).values))

            df_final = pd.concat([x[sig_features],index],axis=1)
            df_final.set_index('CUSTOMER_KEY', inplace=True)
        else:
            # x.set_index('CUSTOMER_KEY', inplace=True)
            df_final=x
            x.reset_index(['CUSTOMER_KEY'], inplace=True)
            index=x['CUSTOMER_KEY']

        return df_final, index

    def predict(models,name,x_score, index):


        ABANDONED={}
        for name, clf in zip(name, models):

            flag=clf.predict(x_score)
            likelihood=clf.predict_proba(x_score)
            defect_prob=[item[0] for item in likelihood]
            retain_prob=[item[1] for item in likelihood]
            ABANDONED[name+'_flag']=flag
            ABANDONED[name+'_retain_prob']=defect_prob
            ABANDONED[name +'_defect_prob' ]=retain_prob
        scored_df=pd.DataFrame.from_dict(ABANDONED)
        scored_df=pd.concat([scored_df, index], axis=1)
        scored_df.set_index('CUSTOMER_KEY', inplace=True)
        return scored_df

                                                    # ***************************TEST************************************************
                                                    # *******************************************************************************
                                                    # *******************************************************************************

    scoring_df, df_cont=import_scoring_data(scoring_data, cont_score_features, bool_score_features,catag_score_features)

    df_no_pca, bool_df, df_cont = import_data(data, cont_features, bool_features, catag_features, scoring_df)



    print 'training dataset loaded..'
    import time
    start_time = time.time()

    print 'scoring dataset loaded..'


    print 'running pca...'
    df_pca, fitted_pca, df_cont = pca_code(df_no_pca,df_cont, do_pca=i)

    print 'running binning...'
    dummy_pca, length_dict = bin_pca(df_pca, bool_df, df_cont, j)

    print 'transforming df to ndarray...'
    x, y = get_arrays(dummy_pca, df_pca, df_no_pca, bool_df, do_pca=i, b_pca=j)
    print 'reducing dimensionality...'
    x, y, plsca= feature_clustering(x, y, fc=k)
    print 'running model algorithm...'
    models, name = algorithm(x, y)
    data = scoring_data_intersection(df_no_pca, scoring_df)
    print list(data)
    print 'transforming scoring data set to pca...'
    df_trans_pca = transform_to_pca(data, fitted_pca, i)
    scoring_df_trans = bin_pca_score_set(df_trans_pca, length_dict, j)
    x = data
    # x = get_scoring_arrays(scoring_df_trans)

    print 'reducing dimensionality for scoring dataset...'
    x_score, index=score_set_feature_selection(x,plsca, k)

    x_score.set_index('CUSTOMER_KEY', inplace=True)
    df_p=predict(models,name,x_score, index)
    # df_predict=pd.concat([df_p,df_index['CUSTOMER_KEY']], axis=1)
    defection=pd.concat([df_p, defection], axis=1)
    print("--- %s seconds ---" % (time.time() - start_time))

                                                    # ***************************END TEST************************************************
                                                    # *******************************************************************************
                                                    # *******************************************************************************



    # file= predict(scoring_df_trans,x_score)
    # a, b, c, d=np.array_split(file, 4)
defection.to_csv(path_or_buf='defection_model_prediction_1.txt', index=True)
    # x_score_a.to_csv(path_or_buf='new_customer_defection_prediction_a.csv', index=False)
    # x_score_b.to_csv(path_or_buf='new_customer_defection_prediction_b.csv', index=False)
    # x_score_c.to_csv(path_or_buf='new_customer_defection_prediction_c.csv', index=False)
    # x_score_d.to_csv(path_or_buf='new_customer_defection_prediction_d.csv', index=False)
    # x_score_e.to_csv(path_or_buf='new_customer_defection_prediction_e.csv', index=False)
    # x_score_f.to_csv(path_or_buf='new_customer_defection_prediction_f.csv', index=False)
    # x_score_g.to_csv(path_or_buf='new_customer_defection_prediction_g.csv', index=False)
    # x_score_h.to_csv(path_or_buf='new_customer_defection_prediction_h.csv', index=False)
    # x_score_i.to_csv(path_or_buf='new_customer_defection_prediction_i.csv', index=False)
    # x_score_j.to_csv(path_or_buf='new_customer_defection_prediction_j.csv', index=False)

    # ***************************TEST************************************************
    # *******************************************************************************
    # *******************************************************************************

