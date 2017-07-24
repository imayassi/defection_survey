import pandas as pd
from sklearn.decomposition import PCA, NMF
from binning import bin
import numpy as np
from statsmodels.tools import categorical
import skfuzzy as fuzz
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
from features_by_customer_type import customer_type_features


from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import pyodbc
import pandas as pd
random_state = np.random.RandomState(0)
ct='New'
cont_features, bool_features, catag_features, _, _, _=customer_type_features(ct)
conn = pyodbc.connect(dsn='VerticaProd')
random_state = np.random.RandomState(0)

data = "SELECT * FROM (SELECT SUP_REMIC_TAXABLE_INCOME,	SUP_REMIC_SCHQ_INCOME,	SUP_REMIC_EXCESS_INCLUSION,	SUP_RE_TOTAL_INCOME,	SUP_RE_PROFESSIONAL_INCOME,	SUP_RE_PERSONAL_DAYS,	SUP_RE_INCOME_ROYALTIES,	SUP_RE_INCOME_RENTS,	SUP_RE_EXPENSES_UTILITIES,	SUP_RE_EXPENSES_TRAVEL,	SUP_RE_EXPENSES_TOTAL,	SUP_RE_EXPENSES_TAXES,	SUP_RE_EXPENSES_SUPPLIES,	SUP_RE_EXPENSES_REPAIRS,	SUP_RE_EXPENSES_OTHER_INTEREST,	SUP_RE_EXPENSES_MORTGAGE_INTEREST,	SUP_RE_EXPENSES_MANAGEMENT,	SUP_RE_EXPENSES_LEGAL,	SUP_RE_EXPENSES_INSURANCE,	SUP_RE_EXPENSES_DEPRECIATION,	SUP_RE_EXPENSES_DEDUCTIBLE_LOSS,	SUP_RE_EXPENSES_COMMISSIONS,	SUP_RE_EXPENSES_CLEANING,	SUP_RE_EXPENSES_ADVERTISING,	SUP_PS_TOTAL_INCOME,	SUP_PS_SEC179_EXPENSE_DEDUCTION,	SUP_PS_PY_LOSS,	SUP_PS_PASSIVE_LOSS,	SUP_PS_PASSIVE_INCOME,	SUP_PS_NONPASSIVE_LOSS,	SUP_PS_NONPASSIVE_INCOME,	SUP_PS_AT_RISK,	SUP_FARM_RENTAL_INCOME,	SUP_FARM_GROSS_INCOME,	SUP_EST_TOTAL_INCOME,	SUP_EST_PASSIVE_LOSS,	SUP_EST_PASSIVE_INCOME,	SUP_EST_NONPASSIVE_LOSS,	SUP_EST_NONPASSIVE_INCOME,	NUM_EXEMPTIONS,	NUM_DEPENDENTS,	FLAG_OLD_OR_BLIND,	BUS_WILL_FILE_1099,	BUS_VEHICLE_MILES_OTHER,	BUS_VEHICLE_MILES_COMMUTE,	BUS_VEHICLE_MILES_BUSINESS,	BUS_STATUTORY_EMPLOYEE,	BUS_START_ACQUIRE,	BUS_REQUIRE_1099,	BUS_OTHER_INCOME,	BUS_NET_PROFIT,	BUS_MATERIAL_PARTICIPATE,	BUS_INVESTMENT_AT_RISK,	BUS_GROSS_PROFIT,	BUS_GROSS_INCOME,	BUS_EXPENSE_WAGES,	BUS_EXPENSE_UTILITIES,	BUS_EXPENSE_USE_OF_HOME,	BUS_EXPENSE_TRAVEL,	BUS_EXPENSE_TOTAL,	BUS_EXPENSE_TAXES,	BUS_EXPENSE_SUPPLIES,	BUS_EXPENSE_REPAIRS,	BUS_EXPENSE_RENT_VEHICLES,	BUS_EXPENSE_RENT_OTHER,	BUS_EXPENSE_PROFIT_TENTATIVE,	BUS_EXPENSE_PENSION,	BUS_EXPENSE_OTHER_INTEREST,	BUS_EXPENSE_OTHER,	BUS_EXPENSE_OFFICE,	BUS_EXPENSE_MORTGAGE_INTEREST,	BUS_EXPENSE_MEALS,	BUS_EXPENSE_LEGAL,	BUS_EXPENSE_INSURANCE,	BUS_EXPENSE_EMPLOYEE_BENEFITS,	BUS_EXPENSE_DEPRECIATION,	BUS_EXPENSE_DEPLETION,	BUS_EXPENSE_CONTRACT_LABOR,	BUS_EXPENSE_COMMISSIONS,	BUS_EXPENSE_CAR,	BUS_EXPENSE_ADVERTISING,	BUS_COGS_TOTAL,	BUS_COGS_SUPPLIES,	BUS_COGS_PURCHASES,	BUS_COGS_LABOR,	BUS_COGS_INVENTORY_START,	BUS_COGS_INVENTORY_END,	BUS_COGS,	AMOUNT_UNREPORTED_SS_MEDICARE_TAX,	AMOUNT_UNEMPLOYMENT,	AMOUNT_TUITION,	AMOUNT_TOTAL_TAX,	AMOUNT_TOTAL_PAYMENTS,	AMOUNT_TOTAL_INTEREST_PAID,	AMOUNT_TOTAL_INCOME,	AMOUNT_TOTAL_DEDUCTIONS,	AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES,	AMOUNT_TOTAL_CREDITS,	AMOUNT_TAXES_PAID,	AMOUNT_TAXABLE_SOCIAL_SEC,	AMOUNT_TAXABLE_OFFSETS,	AMOUNT_TAXABLE_IRA,	AMOUNT_TAXABLE_INTEREST,	AMOUNT_TAXABLE_INCOME,	AMOUNT_TAX_PREP_FEES,	AMOUNT_TAX_DUE,	AMOUNT_TAX_CREDITS,	AMOUNT_TAX,	AMOUNT_STUDENT_LOAN_INTEREST_DEDUCTION,	AMOUNT_STATE_LOCAL_TAX,	AMOUNT_STATE_LOCAL_SALES_TAX,	AMOUNT_SOCIAL_SEC,	AMOUNT_SELF_EMPLOYMENT_TAX,	AMOUNT_SELF_EMPLOYMENT_RETIREMENT,	AMOUNT_SELF_EMPLOYMENT_HEALTH_INSURANCE,	AMOUNT_SCHE,	AMOUNT_SALARIES_AND_WAGES,	AMOUNT_RETIREMENT_SAVINGS_CREDIT,	AMOUNT_RESIDENTIAL_ENERGY_CREDIT,	AMOUNT_REFUND,	AMOUNT_REAL_ESTATE_TAX,	AMOUNT_QUALIFIED_DIVIDENDS,	AMOUNT_PERSONAL_PROPERTY_TAXES,	AMOUNT_PAID_WITH_EXTENSION,	AMOUNT_OTHER_TAXES,	AMOUNT_OTHER_PAYMENTS,	AMOUNT_OTHER_INCOME,	AMOUNT_OTHER_GAIN,	AMOUNT_OTHER_DEDUCTIBLE_TAXES,	AMOUNT_OTHER_DEDUCTIBLE_EXPENSES,	AMOUNT_OTHER_CREDITS,	AMOUNT_ORDINARY_DIVIDENDS,	AMOUNT_NT_COMBAT_PAY,	AMOUNT_MOVING_EXPENSE,	AMOUNT_MORTGAGE_POINTS_NON_1098,	AMOUNT_MORTGAGE_INTEREST_NON_1098,	AMOUNT_MORTGAGE_INTEREST,	AMOUNT_MORTGAGE_INSURANCE,	AMOUNT_MISC_DEDUCTIONS,	AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION,	AMOUNT_MEDICAL_DENTAL_EXPENSES,	AMOUNT_IRA_DISTRIBUTIONS,	AMOUNT_IRA_DEDUCTION,	AMOUNT_INVESTMENT_INTEREST_PAID,	AMOUNT_INCOME_TAX_WITHHELD,	AMOUNT_INCOME_TAX,	AMOUNT_HSA,	AMOUNT_HOPE_CREDIT,	AMOUNT_HOMEBUYER_CREDIT_REPAYMENT,	AMOUNT_FUEL_TAX_CREDIT,	AMOUNT_FOREIGN_TAX_CREDIT,	AMOUNT_FARM_INCOME,	AMOUNT_EXPENSES_DEDUCTION,	AMOUNT_EXEMPTIONS,	AMOUNT_EXCESS_SS_RRTA_WITHHELD,	AMOUNT_ESTIMATED_TAX_PENALTY,	AMOUNT_ESTIMATED_TAX,	AMOUNT_EMPLOYEE_EXPENSES,	AMOUNT_EITC,	AMOUNT_EDUCATION_CREDIT,	AMOUNT_EARLY_WITHDRAWAL_PENALTY,	AMOUNT_DOMESTIC_PRODUCTION_DEDUCTION,	AMOUNT_DISABLED_CREDIT,	AMOUNT_DEDUCTIBLE_SELF_EMPLOYMENT_TAX,	AMOUNT_CHILD_CREDIT,	AMOUNT_CHILD_CARE_CREDIT,	AMOUNT_CHARITABLE_CONTRIBUTIONS_NONCASH,	AMOUNT_CHARITABLE_CONTRIBUTIONS_CASH,	AMOUNT_CHARITABLE_CONTRIBUTIONS_CARRYOVER,	AMOUNT_CHARITABLE_CONTRIBUTIONS,	AMOUNT_CERTAIN_BUSINESS_EXPENSE,	AMOUNT_CASUALTY_LOSSES,	AMOUNT_CAPITAL_GAIN,	AMOUNT_BUSINESS_INCOME,	AMOUNT_AMT,	AMOUNT_ALIMONY_PAID,	AMOUNT_ALIMONY_INCOME,	AMOUNT_ADJUSTMENTS,	AGI,	AGE_TAXPAYER,	AGE_SPOUSE,	AGE_DEPENDENT_MIN,	AGE_DEPENDENT_MAX,	AGE_DEPENDENT_AVG,	CUSTOMER_KEY,	ABANDONED,	TTO_FLAG,	STATE_ATTACH_COUNT,	MINDBENDER_FLAG,	CA_REFUND_TRANSFER_FLAG,	NON_CA_REFUND_TRANSFER_FLAG,	REFUND_TRANSFER_FLAG,	CA_AUDIT_DEFENSE_FLAG,	NON_CA_AUDIT_DEFENSE_FLAG,	AUDIT_DEFENSE_FLAG,	CA_MAX_FLAG,	NON_CA_MAX_FLAG,	MAX_FLAG,	PS_FLAG,	PRS_SCORE,	REJECT_COUNT,	TAX_WEEK,	TAX_DAY,	CORE_FLAG,	REQUIRED_TAKE_FLAG,	FSCHC_FLAG,	FSCHCEZ_FLAG,	FSCHE_FLAG,	FSCHA_FLAG,	FSCHD_FLAG,	FSCHF_FLAG,	MISC1099_FLAG,	ACCEPTED_EFILE,	NUM_CARE_CONTACTS,	RT_FLAG,	PRE_UPVOTES,	PRE_DOWNVOTES,	PRE_TOTAL_VOTES,	POST_UPVOTES,	POST_DOWNVOTES,	POST_TOTAL_VOTES,	SUP_RE_WILL_FILE_1099,	SUP_RE_TYPE_SINGLE_FAMILY,	SUP_RE_TYPE_SHORT_RENTAL,	SUP_RE_TYPE_SELF_RENTAL,	SUP_RE_TYPE_ROYALTIES,	SUP_RE_TYPE_OTHER,	SUP_RE_TYPE_MULTI_FAMILY,	SUP_RE_TYPE_LAND,	SUP_RE_TYPE_COMMERCIAL,	SUP_RE_REQUIRE_1099,	SUP_PS_SCORP,	SUP_PS_PARTNERSHIP_FOREIGN,	SUP_PS_PARTNERSHIP,	NUM_W2,	NUM_SCHE,	NUM_SCHC,	FLAG_ITEMIZED_DEDUCTIONS,	BUS_VEHICLE_PERSONAL_OFFDUTY,	BUS_VEHICLE_PERSONAL_ANOTHER,	BUS_COGS_METHOD_OTHER,	BUS_COGS_METHOD_COST_OR_MARKET,	BUS_COGS_METHOD_COST,	BUS_ACCOUNTING_METHOD_OTHER,	BUS_ACCOUNTING_METHOD_CASH,	BUS_ACCOUNTING_METHOD_ACCRUAL,	student_taxpayer,	_taxpayer,	retired_taxpayer,	unemployed_taxpayer,	cashier_taxpayer,	teacher_taxpayer,	manager_taxpayer,	sales_taxpayer,	customerservice_taxpayer,	laborer_taxpayer,	cook_taxpayer,	server_taxpayer,	retail_taxpayer,	engineer_taxpayer,	military_taxpayer,	cna_taxpayer,	salesassociate_taxpayer,	registerednurse_taxpayer,	truckdriver_taxpayer,	driver_taxpayer,	mechanic_taxpayer,	nurse_taxpayer,	waitress_taxpayer,	supervisor_taxpayer,	labor_taxpayer,	construction_taxpayer,	warehouse_taxpayer,	administrativeassistant_taxpayer,	accountant_taxpayer,	customerservicerep_taxpayer,	clerk_taxpayer,	receptionist_taxpayer,	medicalassistant_taxpayer,	disabled_taxpayer,	softwareengineer_taxpayer,	assistantmanager_taxpayer,	officemanager_taxpayer,	electrician_taxpayer,	technician_taxpayer,	none_taxpayer,	machineoperator_taxpayer,	caregiver_taxpayer,	employed_taxpayer,	bartender_taxpayer,	projectmanager_taxpayer,	maintenance_taxpayer,	welder_taxpayer,	housekeeper_taxpayer,	homemaker_taxpayer,	foodservice_taxpayer,	selfemployed_taxpayer,	chef_taxpayer,	csr_taxpayer,	warehouseworker_taxpayer,	factoryworker_taxpayer,	attorney_taxpayer,	consultant_taxpayer,	policeofficer_taxpayer,	carpenter_taxpayer,	secretary_taxpayer,	rn_taxpayer,	banker_taxpayer,	operator_taxpayer,	worker_taxpayer,	security_taxpayer,	production_taxpayer,	socialworker_taxpayer,	securityofficer_taxpayer,	pharmacytechnician_taxpayer,	machinist_taxpayer,	barista_taxpayer,	analyst_taxpayer,	deliverydriver_taxpayer,	stocker_taxpayer,	factory_taxpayer,	dentalassistant_taxpayer,	custodian_taxpayer,	generalmanager_taxpayer,	employee_taxpayer,	salesmanager_taxpayer,	retailmanager_taxpayer,	housekeeping_taxpayer,	storemanager_taxpayer,	securityguard_taxpayer,	na_taxpayer,	painter_taxpayer,	accountmanager_taxpayer,	correctionalofficer_taxpayer,	lpn_taxpayer,	operationsmanager_taxpayer,	bankteller_taxpayer,	fastfood_taxpayer,	manufacturing_taxpayer,	crewmember_taxpayer,	generallabor_taxpayer,	plumber_taxpayer,	marketing_taxpayer,	dispatcher_taxpayer,	hostess_taxpayer,	management_taxpayer,	janitor_taxpayer,	homemaker_spouse,	retired_spouse,	unemployed_spouse,	teacher_spouse,	student_spouse,	housewife_spouse,	none_spouse,	disabled_spouse,	home_maker_spouse,	house_wife_spouse,	stayathomemom_spouse,	nurse_spouse,	manager_spouse,	registerednurse_spouse,	sales_spouse,	customerservice_spouse,	cashier_spouse,	na_spouse,	laborer_spouse,	retail_spouse,	officemanager_spouse,	administrativeassistant_spouse,	engineer_spouse,	secretary_spouse,	accountant_spouse,	cook_spouse,	receptionist_spouse,	truckdriver_spouse,	self_employed_spouse,	mechanic_spouse,	rn_spouse,	server_spouse,	cna_spouse,	driver_spouse,	supervisor_spouse,	medicalassistant_spouse,	clerk_spouse,	socialworker_spouse,	projectmanager_spouse,	salesassociate_spouse,	mother_spouse,	waitress_spouse,	military_spouse,	construction_spouse,	attorney_spouse,	electrician_spouse,	dentalassistant_spouse,	stayathomemother_spouse,	customerservicerep_spouse,	mom_spouse,	substituteteacher_spouse,	consultant_spouse,	maintenance_spouse,	policeofficer_spouse,	banker_spouse,	labor_spouse,	welder_spouse,	softwareengineer_spouse,	na_spouse2,	technician_spouse,	marketing_spouse,	assistantmanager_spouse,	bankteller_spouse,	stayathomeparent_spouse,	caregiver_spouse,	warehouse_spouse,	educator_spouse,	carpenter_spouse,	bookkeeper_spouse,	housekeeper_spouse,	physician_spouse,	clerical_spouse,	analyst_spouse,	paralegal_spouse,	pharmacist_spouse,	machineoperator_spouse,	chef_spouse,	physicaltherapist_spouse,	humanresources_spouse,	lpn_spouse,	foodservice_spouse,	pharmacytechnician_spouse,	executiveassistant_spouse,	officeassistant_spouse,	disable_spouse,	salesmanager_spouse,	accountmanager_spouse,	professor_spouse,	hairstylist_spouse,	machinist_spouse,	selfemployed_spouse,	custodian_spouse,	factoryworker_spouse,	retailmanager_spouse,	director_spouse,	graphicdesigner_spouse,	bartender_spouse,	administrator_spouse,	home_spouse,	accounting_spouse,	unemployeed_spouse,	ORDER_AMOUNT,	FEDERAL_REVENUE,	MINDBENDER_REVENUE,	CA_REFUND_TRANSFER_REVENUE,	NON_CA_REFUND_TRANSFER_REVENUE,	REFUND_TRANSFER_REVENUE,	CA_AUDIT_DEFENSE_REVENUE,	NON_CA_AUDIT_DEFENSE_REVENUE,	AUDIT_DEFENSE_REVENUE,	CA_MAX_REVENUE,	NON_CA_MAX_REVENUE,	MAX_REVENUE,	PS_REVENUE,	FLAG_ITEMIZE_SEPARATELY,	TOTAL_REVENUE,	STATE_REVENUE,	START_SKU,	ENTRY_PAGE_GROUP,	CHANNEL,	IMPORT_TYPE,	START_SKU_ROLLUP,	FED_FORM_TYPE,	PRODUCT_ROLLUP,	PRODUCT_EDITION_DESCRIPTION,	CUSTOMER_TYPE_ROLLUP,	CUSTOMER_TYPE,	CUSTOMER_DEFINITION,	CUSTOMER_DEFINITION_ADJ,	NEW_CUSTOMER_DEFINITION,	COMPLETED_SKU,	LAST_STATUS,	FILING_STATUS,	DMA_AREA FROM  CTG_ANALYTICS_WS.SM_RETENTION_MODEL  WHERE   TAX_YEAR<2014 AND TAX_DAY<=150)A   order by random() limit 100000"
df = pd.read_sql(data, conn, index_col=['CUSTOMER_KEY'], coerce_float=False)
print tuple(list(df))
df_cont = df[cont_features]
df_cont.columns = df_cont.columns.str.strip()
df_cont.fillna(value=0, inplace=True)
df_cont.replace(to_replace=('(null)', 'NA', 'None'), value=0)
df_bool = df_cont[bool_features]
df_cont.drop(bool_features, axis=1, inplace=True)
index_df = pd.DataFrame(df_cont.reset_index(level=['CUSTOMER_KEY']), columns=['CUSTOMER_KEY'])

data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
data_scaled = pd.concat([data_scaled, index_df], axis=1)
data_scaled.set_index('CUSTOMER_KEY', inplace=True)

df_char = df[catag_features]
df_char.columns = df_char.columns.str.strip()
df_char.fillna(value='-1', inplace=True)
df_char.replace(to_replace=('(null)', 'NA', 'None'), value='-1')
just_dummies = pd.get_dummies(df_char)


df_trans = pd.concat([df_bool, just_dummies, data_scaled], axis=1)
df_trans.reset_index(['CUSTOMER_KEY'], inplace=True)

y=df_trans[['CUSTOMER_KEY','ABANDONED']]
y.set_index('CUSTOMER_KEY', inplace=True)
x=df_trans.drop(['ABANDONED'], axis=1)
x.set_index('CUSTOMER_KEY', inplace=True)


X=x
train_sizes, train_scores, valid_scores = learning_curve(RandomForestClassifier(criterion='entropy', n_estimators=200), X, y, train_sizes=[50000, 100000, 500000, 1000000], cv=3)