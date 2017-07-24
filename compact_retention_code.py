import time

start_time = time.time()
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
# import vertica_python
import pyodbc
import pandas as pd
from pandas import Series,DataFrame
# from retention_customer_type_features_new import customer_type_features
from features_by_customer_type import customer_type_features
conn = pyodbc.connect(dsn='VerticaProd')

defection=pd.DataFrame()
random_state = np.random.RandomState(0)
i='False'
j='False'
ct='Loyal'
# cont_features, bool_features, catag_features, cont_score_features, bool_score_features, catag_score_features=customer_type_features(ct)
cont_features=[	'ABANDONED',	'AGE_DEPENDENT_MAX',	'AGE_DEPENDENT_MAX_PY',	'AGE_DEPENDENT_MAX_PY2',	'AGE_DEPENDENT_MIN_PY',	'AGE_DEPENDENT_MIN_PY2',	'AGE_SPOUSE_PY2',	'AGE_TAXPAYER_PY',	'AGE_TAXPAYER_PY2',	'AMOUNT_ADJUSTMENTS',	'AMOUNT_CHILD_CREDIT',	'AMOUNT_EDUCATION_CREDIT_PY2',	'AMOUNT_EXEMPTIONS_PY',	'AMOUNT_INCOME_TAX_WITHHELD_PY',	'AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION_PY2',	'AMOUNT_ORDINARY_DIVIDENDS_PY',	'AMOUNT_REFUND',	'AMOUNT_SALARIES_AND_WAGES_PY',	'AMOUNT_SCHE_PY',	'AMOUNT_SCHE_PY2',	'AMOUNT_SOCIAL_SEC_PY2',	'AMOUNT_TAX_DUE_PY',	'AMOUNT_TAXABLE_INCOME',	'AMOUNT_TAXABLE_SOCIAL_SEC_PY',	'AMOUNT_TOTAL_CREDITS_PY',	'AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES_PY',	'AUDIT_DEFENSE_FLAG',	'AUTH_NOT_COMPLETE_PY2',	'AUTH_NOT_COMPLETE_PY3',	'BUS_COGS_TOTAL_PY2',	'BUS_EXPENSE_EMPLOYEE_BENEFITS_PY2',	'BUS_GROSS_INCOME_PY3',	'BUS_NET_PROFIT',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'NON_CA_REFUND_TRANSFER_REVENUE',	'NUM_W2_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'SUP_PS_AT_RISK',	'SUP_PS_AT_RISK_PY2',	'SUP_PS_PARTNERSHIP_PY',	'SUP_PS_PARTNERSHIP_PY2',	'SUP_RE_EXPENSES_TOTAL',	'SUP_RE_EXPENSES_TOTAL_PY3',	'SUP_RE_INCOME_RENTS',	'SUP_RE_TYPE_ROYALTIES_PY3',	'SUP_RE_TYPE_SELF_RENTAL_PY',	'TTO_FLAG_PY2']
bool_features=['ABANDONED',	'AUDIT_DEFENSE_FLAG',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'TTO_FLAG_PY2']
catag_features=['TAX_DAY','TAX_DAY_PY','TAX_DAY_PY2','TAX_DAY_PY3','CHANNEL',	'CHANNEL_PY',	'COMPLETED_SKU',	'COMPLETED_SKU_PY',	'COMPLETED_SKU_PY2',	'COMPLETED_SKU_PY3',	'CUSTOMER_DEFINITION_ADJ',	'CUSTOMER_DEFINITION_ADJ_PY',	'CUSTOMER_DEFINITION_ADJ_PY2',	'ENTRY_PAGE_GROUP',	'ENTRY_PAGE_GROUP_PY',	'FED_FORM_TYPE',	'FED_FORM_TYPE_PY',	'FED_FORM_TYPE_PY2',	'FILING_STATUS',		'FILING_STATUS_PY',	'FILING_STATUS_PY2',	'FILING_STATUS_PY3',	'IMPORT_TYPE',	'IMPORT_TYPE_PY',	'IMPORT_TYPE_PY2',	'LAST_STATUS',	'LAST_STATUS_PY3',	'PRODUCT_EDITION_DESCRIPTION',	'PRODUCT_EDITION_DESCRIPTION_PY3',	'PRODUCT_ROLLUP',	'PRODUCT_ROLLUP_PY2',	'PRODUCT_ROLLUP_PY3',	'START_SKU_PY',	'START_SKU_PY2',	'START_SKU_ROLLUP_PY']
cont_score_features=[	'AGE_DEPENDENT_MAX',	'AGE_DEPENDENT_MAX_PY',	'AGE_DEPENDENT_MAX_PY2',	'AGE_DEPENDENT_MIN_PY',	'AGE_DEPENDENT_MIN_PY2',	'AGE_SPOUSE_PY2',	'AGE_TAXPAYER_PY',	'AGE_TAXPAYER_PY2',	'AMOUNT_ADJUSTMENTS',	'AMOUNT_CHILD_CREDIT',	'AMOUNT_EDUCATION_CREDIT_PY2',	'AMOUNT_EXEMPTIONS_PY',	'AMOUNT_INCOME_TAX_WITHHELD_PY',	'AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION_PY2',	'AMOUNT_ORDINARY_DIVIDENDS_PY',	'AMOUNT_REFUND',	'AMOUNT_SALARIES_AND_WAGES_PY',	'AMOUNT_SCHE_PY',	'AMOUNT_SCHE_PY2',	'AMOUNT_SOCIAL_SEC_PY2',	'AMOUNT_TAX_DUE_PY',	'AMOUNT_TAXABLE_INCOME',	'AMOUNT_TAXABLE_SOCIAL_SEC_PY',	'AMOUNT_TOTAL_CREDITS_PY',	'AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES_PY',	'AUDIT_DEFENSE_FLAG',	'AUTH_NOT_COMPLETE_PY2',	'AUTH_NOT_COMPLETE_PY3',	'BUS_COGS_TOTAL_PY2',	'BUS_EXPENSE_EMPLOYEE_BENEFITS_PY2',	'BUS_GROSS_INCOME_PY3',	'BUS_NET_PROFIT',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'NON_CA_REFUND_TRANSFER_REVENUE',	'NUM_W2_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'SUP_PS_AT_RISK',	'SUP_PS_AT_RISK_PY2',	'SUP_PS_PARTNERSHIP_PY',	'SUP_PS_PARTNERSHIP_PY2',	'SUP_RE_EXPENSES_TOTAL',	'SUP_RE_EXPENSES_TOTAL_PY3',	'SUP_RE_INCOME_RENTS',	'SUP_RE_TYPE_ROYALTIES_PY3',	'SUP_RE_TYPE_SELF_RENTAL_PY',	'TTO_FLAG_PY2']
bool_score_features=['AUDIT_DEFENSE_FLAG',	'FLAG_ITEMIZED_DEDUCTIONS_PY',	'FLAG_OLD_OR_BLIND',	'FLAG_OLD_OR_BLIND_PY',	'FSCHA_FLAG',	'FSCHE_FLAG',	'MISC1099_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG',	'NON_CA_REFUND_TRANSFER_FLAG_PY2',	'REFUND_TRANSFER_FLAG',	'REFUND_TRANSFER_FLAG_PY3',	'REQUIRED_TAKE_FLAG',	'REQUIRED_TAKE_FLAG_PY',	'RISK_FLAG',	'TTO_FLAG_PY2']
catag_score_features=['TAX_DAY','TAX_DAY_PY','TAX_DAY_PY2','TAX_DAY_PY3','CHANNEL',	'CHANNEL_PY',	'COMPLETED_SKU',	'COMPLETED_SKU_PY',	'COMPLETED_SKU_PY2',	'COMPLETED_SKU_PY3',	'CUSTOMER_DEFINITION_ADJ',	'CUSTOMER_DEFINITION_ADJ_PY',	'CUSTOMER_DEFINITION_ADJ_PY2',	'ENTRY_PAGE_GROUP',	'ENTRY_PAGE_GROUP_PY',	'FED_FORM_TYPE',	'FED_FORM_TYPE_PY',	'FED_FORM_TYPE_PY2',	'FILING_STATUS',	'FILING_STATUS_PY',	'FILING_STATUS_PY2',	'FILING_STATUS_PY3',	'IMPORT_TYPE',	'IMPORT_TYPE_PY',	'IMPORT_TYPE_PY2',	'LAST_STATUS',	'LAST_STATUS_PY3',	'PRODUCT_EDITION_DESCRIPTION',	'PRODUCT_EDITION_DESCRIPTION_PY3',	'PRODUCT_ROLLUP',	'PRODUCT_ROLLUP_PY2',	'PRODUCT_ROLLUP_PY3',	'START_SKU_PY',	'START_SKU_PY2',	'START_SKU_ROLLUP_PY']


# scoring_data="SELECT * FROM (SELECT CUSTOMER_KEY,TAX_DAY,TAX_DAY_PY,TAX_DAY_PY2,TAX_DAY_PY3,		AGE_DEPENDENT_MAX,	AGE_DEPENDENT_MAX_PY,	AGE_DEPENDENT_MAX_PY2,	AGE_DEPENDENT_MIN_PY,	AGE_DEPENDENT_MIN_PY2,	AGE_SPOUSE_PY2,	AGE_TAXPAYER_PY,	AGE_TAXPAYER_PY2,	AMOUNT_ADJUSTMENTS,	AMOUNT_CHILD_CREDIT,	AMOUNT_EDUCATION_CREDIT_PY2,	AMOUNT_EXEMPTIONS_PY,	AMOUNT_INCOME_TAX_WITHHELD_PY,	AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION_PY2,	AMOUNT_ORDINARY_DIVIDENDS_PY,	AMOUNT_REFUND,	AMOUNT_SALARIES_AND_WAGES_PY,	AMOUNT_SCHE_PY,	AMOUNT_SCHE_PY2,	AMOUNT_SOCIAL_SEC_PY2,	AMOUNT_TAX_DUE_PY,	AMOUNT_TAXABLE_INCOME,	AMOUNT_TAXABLE_SOCIAL_SEC_PY,	AMOUNT_TOTAL_CREDITS_PY,	AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES_PY,	AUDIT_DEFENSE_FLAG,	AUTH_NOT_COMPLETE_PY2,	AUTH_NOT_COMPLETE_PY3,	BUS_COGS_TOTAL_PY2,	BUS_EXPENSE_EMPLOYEE_BENEFITS_PY2,	BUS_GROSS_INCOME_PY3,	BUS_NET_PROFIT,	CHANNEL,	CHANNEL_PY,	COMPLETED_SKU,	COMPLETED_SKU_PY,	COMPLETED_SKU_PY2,	COMPLETED_SKU_PY3,	CUSTOMER_DEFINITION_ADJ,	CUSTOMER_DEFINITION_ADJ_PY,		CUSTOMER_DEFINITION_ADJ_PY2,	ENTRY_PAGE_GROUP,	ENTRY_PAGE_GROUP_PY,	FED_FORM_TYPE,	FED_FORM_TYPE_PY,	FED_FORM_TYPE_PY2,	FILING_STATUS,	FILING_STATUS_PY,	FILING_STATUS_PY2,	FILING_STATUS_PY3,	FLAG_ITEMIZED_DEDUCTIONS_PY,	FLAG_OLD_OR_BLIND,	FLAG_OLD_OR_BLIND_PY,	FSCHA_FLAG,	FSCHE_FLAG,	IMPORT_TYPE,	IMPORT_TYPE_PY,	IMPORT_TYPE_PY2,	LAST_STATUS,	LAST_STATUS_PY3,	MISC1099_FLAG,	NON_CA_REFUND_TRANSFER_FLAG,	NON_CA_REFUND_TRANSFER_FLAG_PY2,	NON_CA_REFUND_TRANSFER_REVENUE,	NUM_W2_PY2,	PRODUCT_EDITION_DESCRIPTION,	PRODUCT_EDITION_DESCRIPTION_PY3,	PRODUCT_ROLLUP,	PRODUCT_ROLLUP_PY2,	PRODUCT_ROLLUP_PY3,	REFUND_TRANSFER_FLAG,	REFUND_TRANSFER_FLAG_PY3,	REQUIRED_TAKE_FLAG,	REQUIRED_TAKE_FLAG_PY,	RISK_FLAG,	START_SKU_PY,	START_SKU_PY2,	START_SKU_ROLLUP_PY,	SUP_PS_AT_RISK,	SUP_PS_AT_RISK_PY2,	SUP_PS_PARTNERSHIP_PY,	SUP_PS_PARTNERSHIP_PY2,	SUP_RE_EXPENSES_TOTAL,	SUP_RE_EXPENSES_TOTAL_PY3,	SUP_RE_INCOME_RENTS,	SUP_RE_TYPE_ROYALTIES_PY3,	SUP_RE_TYPE_SELF_RENTAL_PY,	TTO_FLAG_PY2, ROW_NUMBER() OVER() AS RANK2 FROM  CTG_ANALYTICS_WS.SM_RETENTION_MODEL  WHERE  TAX_YEAR=2016)A  WHERE RANK2 between 1 and 300000"
data = "SELECT * FROM (SELECT CUSTOMER_KEY,TAX_DAY,TAX_DAY_PY,TAX_DAY_PY2,TAX_DAY_PY3,	ABANDONED,	AGE_DEPENDENT_MAX,	AGE_DEPENDENT_MAX_PY,	AGE_DEPENDENT_MAX_PY2,	AGE_DEPENDENT_MIN_PY,	AGE_DEPENDENT_MIN_PY2,	AGE_SPOUSE_PY2,	AGE_TAXPAYER_PY,	AGE_TAXPAYER_PY2,	AMOUNT_ADJUSTMENTS,	AMOUNT_CHILD_CREDIT,	AMOUNT_EDUCATION_CREDIT_PY2,	AMOUNT_EXEMPTIONS_PY,	AMOUNT_INCOME_TAX_WITHHELD_PY,	AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION_PY2,	AMOUNT_ORDINARY_DIVIDENDS_PY,	AMOUNT_REFUND,	AMOUNT_SALARIES_AND_WAGES_PY,	AMOUNT_SCHE_PY,	AMOUNT_SCHE_PY2,	AMOUNT_SOCIAL_SEC_PY2,	AMOUNT_TAX_DUE_PY,	AMOUNT_TAXABLE_INCOME,	AMOUNT_TAXABLE_SOCIAL_SEC_PY,	AMOUNT_TOTAL_CREDITS_PY,	AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES_PY,	AUDIT_DEFENSE_FLAG,	AUTH_NOT_COMPLETE_PY2,	AUTH_NOT_COMPLETE_PY3,	BUS_COGS_TOTAL_PY2,	BUS_EXPENSE_EMPLOYEE_BENEFITS_PY2,	BUS_GROSS_INCOME_PY3,	BUS_NET_PROFIT,	CHANNEL,	CHANNEL_PY,	COMPLETED_SKU,	COMPLETED_SKU_PY,	COMPLETED_SKU_PY2,	COMPLETED_SKU_PY3,	CUSTOMER_DEFINITION_ADJ,	CUSTOMER_DEFINITION_ADJ_PY,	CUSTOMER_DEFINITION_ADJ_PY2,	ENTRY_PAGE_GROUP,	ENTRY_PAGE_GROUP_PY,	FED_FORM_TYPE,	FED_FORM_TYPE_PY,	FED_FORM_TYPE_PY2,	FILING_STATUS,	FILING_STATUS_PY,	FILING_STATUS_PY2,	FILING_STATUS_PY3,	FLAG_ITEMIZED_DEDUCTIONS_PY,	FLAG_OLD_OR_BLIND,	FLAG_OLD_OR_BLIND_PY,	FSCHA_FLAG,	FSCHE_FLAG,	IMPORT_TYPE,	IMPORT_TYPE_PY,	IMPORT_TYPE_PY2,	LAST_STATUS,	LAST_STATUS_PY3,	MISC1099_FLAG,	NON_CA_REFUND_TRANSFER_FLAG,	NON_CA_REFUND_TRANSFER_FLAG_PY2,	NON_CA_REFUND_TRANSFER_REVENUE,	NUM_W2_PY2,	PRODUCT_EDITION_DESCRIPTION,	PRODUCT_EDITION_DESCRIPTION_PY3,	PRODUCT_ROLLUP,	PRODUCT_ROLLUP_PY2,	PRODUCT_ROLLUP_PY3,	REFUND_TRANSFER_FLAG,	REFUND_TRANSFER_FLAG_PY3,	REQUIRED_TAKE_FLAG,	REQUIRED_TAKE_FLAG_PY,	RISK_FLAG,	START_SKU_PY,	START_SKU_PY2,	START_SKU_ROLLUP_PY,	SUP_PS_AT_RISK,	SUP_PS_AT_RISK_PY2,	SUP_PS_PARTNERSHIP_PY,	SUP_PS_PARTNERSHIP_PY2,	SUP_RE_EXPENSES_TOTAL,	SUP_RE_EXPENSES_TOTAL_PY3,	SUP_RE_INCOME_RENTS,	SUP_RE_TYPE_ROYALTIES_PY3,	SUP_RE_TYPE_SELF_RENTAL_PY,	TTO_FLAG_PY2 FROM  CTG_ANALYTICS_WS.SM_RETENTION_MODEL  WHERE   TAX_YEAR<2015  )A   order by random() limit 100000"



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
print list(df_trans)
names = ["Random_Forest"]
Y = df_bool['ABANDONED']
X = df_trans.drop(['ABANDONED'], axis=1)
y = Y
x = X

# plsca = PLSRegression(n_components=50)
# plsca.fit(x, y)
# x3 = plsca.transform(x)
# string = "pls_"
# pls_column_name = [string + `i` for i in range(x3.shape[1])]
# df1 = pd.DataFrame(x3, columns=pls_column_name)
# plsca_df = pd.DataFrame(plsca.x_weights_)
# plsca_trans = plsca_df.transpose()
# x.reset_index(['CUSTOMER_KEY'], inplace=True)
# x.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
# reduced_df = pd.DataFrame(plsca_trans.values, columns=x.columns, index=pls_column_name)
# sig_features = list(set(reduced_df.idxmax(axis=1).values))
# print sig_features
# df_final = x[sig_features]
# bool = pd.DataFrame(bool_df['ABANDONED'], columns=['ABANDONED'])
# bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)
# df = pd.concat([df_final, bool['ABANDONED']], axis=1)

# x=df_final
# y=bool['ABANDONED']
classifiers = [RandomForestClassifier(criterion='entropy', n_estimators=200)]
models = []
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.4, random_state=np.random.RandomState(0))
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision = average_precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    f1 = f1_score(y_test, y_pred)
    # f1_avg=np.mean(cross_val_score(clf, X_train, y_train, scoring="f1", cv=3, n_jobs=1))
    # auc = np.mean(cross_val_score(clf, X_train, y_train, scoring="roc_auc", cv=3, n_jobs=1))
    # precision_avg = np.mean(cross_val_score(clf, X_train, y_train, scoring="precision", cv=3, n_jobs=1))
    # recall_avg = np.mean(cross_val_score(clf, X_train, y_train, scoring="recall", cv=3, n_jobs=1))
    print name, precision, recall, f1, auc, tn, fp, fn, tp
    # , "recall_avg ", recall_avg, "precision_avg ", precision_avg

    models.append(clf)
    # return name, ' model', ' precision score', precision, ' recall score', recall, ' f1 ', f1

scoring_data = "SELECT A.* FROM (SELECT CUSTOMER_KEY,TAX_DAY,TAX_DAY_PY,TAX_DAY_PY2,TAX_DAY_PY3,	ABANDONED,	AGE_DEPENDENT_MAX,	AGE_DEPENDENT_MAX_PY,	AGE_DEPENDENT_MAX_PY2,	AGE_DEPENDENT_MIN_PY,	AGE_DEPENDENT_MIN_PY2,	AGE_SPOUSE_PY2,	AGE_TAXPAYER_PY,	AGE_TAXPAYER_PY2,	AMOUNT_ADJUSTMENTS,	AMOUNT_CHILD_CREDIT,	AMOUNT_EDUCATION_CREDIT_PY2,	AMOUNT_EXEMPTIONS_PY,	AMOUNT_INCOME_TAX_WITHHELD_PY,	AMOUNT_MEDICAL_DENTAL_EXPENSES_DEDUCTION_PY2,	AMOUNT_ORDINARY_DIVIDENDS_PY,	AMOUNT_REFUND,	AMOUNT_SALARIES_AND_WAGES_PY,	AMOUNT_SCHE_PY,	AMOUNT_SCHE_PY2,	AMOUNT_SOCIAL_SEC_PY2,	AMOUNT_TAX_DUE_PY,	AMOUNT_TAXABLE_INCOME,	AMOUNT_TAXABLE_SOCIAL_SEC_PY,	AMOUNT_TOTAL_CREDITS_PY,	AMOUNT_TOTAL_DEDUCTIBLE_EXPENSES_PY,	AUDIT_DEFENSE_FLAG,	AUTH_NOT_COMPLETE_PY2,	AUTH_NOT_COMPLETE_PY3,	BUS_COGS_TOTAL_PY2,	BUS_EXPENSE_EMPLOYEE_BENEFITS_PY2,	BUS_GROSS_INCOME_PY3,	BUS_NET_PROFIT,	CHANNEL,	CHANNEL_PY,	COMPLETED_SKU,	COMPLETED_SKU_PY,	COMPLETED_SKU_PY2,	COMPLETED_SKU_PY3,	CUSTOMER_DEFINITION_ADJ,	CUSTOMER_DEFINITION_ADJ_PY,	CUSTOMER_DEFINITION_ADJ_PY2,	ENTRY_PAGE_GROUP,	ENTRY_PAGE_GROUP_PY,	FED_FORM_TYPE,	FED_FORM_TYPE_PY,	FED_FORM_TYPE_PY2,	FILING_STATUS,	FILING_STATUS_PY,	FILING_STATUS_PY2,	FILING_STATUS_PY3,	FLAG_ITEMIZED_DEDUCTIONS_PY,	FLAG_OLD_OR_BLIND,	FLAG_OLD_OR_BLIND_PY,	FSCHA_FLAG,	FSCHE_FLAG,	IMPORT_TYPE,	IMPORT_TYPE_PY,	IMPORT_TYPE_PY2,	LAST_STATUS,	LAST_STATUS_PY3,	MISC1099_FLAG,	NON_CA_REFUND_TRANSFER_FLAG,	NON_CA_REFUND_TRANSFER_FLAG_PY2,	NON_CA_REFUND_TRANSFER_REVENUE,	NUM_W2_PY2,	PRODUCT_EDITION_DESCRIPTION,	PRODUCT_EDITION_DESCRIPTION_PY3,	PRODUCT_ROLLUP,	PRODUCT_ROLLUP_PY2,	PRODUCT_ROLLUP_PY3,	REFUND_TRANSFER_FLAG,	REFUND_TRANSFER_FLAG_PY3,	REQUIRED_TAKE_FLAG,	REQUIRED_TAKE_FLAG_PY,	RISK_FLAG,	START_SKU_PY,	START_SKU_PY2,	START_SKU_ROLLUP_PY,	SUP_PS_AT_RISK,	SUP_PS_AT_RISK_PY2,	SUP_PS_PARTNERSHIP_PY,	SUP_PS_PARTNERSHIP_PY2,	SUP_RE_EXPENSES_TOTAL,	SUP_RE_EXPENSES_TOTAL_PY3,	SUP_RE_INCOME_RENTS,	SUP_RE_TYPE_ROYALTIES_PY3,	SUP_RE_TYPE_SELF_RENTAL_PY,	TTO_FLAG_PY2 FROM  CTG_ANALYTICS_WS.SM_RETENTION_MODEL  WHERE   TAX_YEAR=2016  )A inner join (SELECT DISTINCT CUSTOMER_KEY FROM  CTG_ANALYTICS_WS.SM_RETENTION_MODEL  WHERE   TAX_YEAR=2016 AND TAX_DAY>=112) B ON A.CUSTOMER_KEY=B.CUSTOMER_KEY "

df = pd.read_sql(scoring_data, conn, index_col='CUSTOMER_KEY', coerce_float=False)
# df = df_raw[df_raw['TAX_YEAR'] == 2016]

df_cont = df[cont_score_features]
# print df_cont

df_cont.columns = df_cont.columns.str.strip()
df_cont.fillna(value=0, inplace=True)
df_cont.replace(to_replace=('(null)', 'NA', 'None'), value=0)
df_cont = df_cont.astype(float)

df_bool = df_cont[bool_score_features]
df_cont.drop(bool_score_features, axis=1, inplace=True)

index_df = pd.DataFrame(df_cont.reset_index(level=['CUSTOMER_KEY']), columns=['CUSTOMER_KEY'])

data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
data_scaled = pd.concat([data_scaled, index_df], axis=1)
data_scaled.set_index('CUSTOMER_KEY', inplace=True)

df_char = df[catag_score_features]
df_char = df_char.astype(object)
print
df_char.dtypes

df_char.columns = df_char.columns.str.strip()
df_char.fillna(value='-1', inplace=True)
df_char.replace(to_replace=('(null)', 'NA', 'None', '', ' ', '\t'), value='-1')

# print list(df_char)
# print df_char
#
just_dummies = pd.get_dummies(df_char)
# print 'SCORING...', just_dummies
df_trans = pd.concat([df_bool, just_dummies, data_scaled], axis=1)
print list(df_trans)
ABANDONED = {}
for name, clf in zip(name, models):
    flag = clf.predict(df_trans)
    likelihood = clf.predict_proba(df_trans)
    defect_prob = [item[0] for item in likelihood]
    retain_prob = [item[1] for item in likelihood]
    ABANDONED[name + '_flag'] = flag
    ABANDONED[name + '_retain_prob'] = defect_prob
    ABANDONED[name + '_defect_prob'] = retain_prob
scored_df = pd.DataFrame.from_dict(ABANDONED)
scored_df = pd.concat([scored_df, index], axis=1)
scored_df.set_index('CUSTOMER_KEY', inplace=True)



print("--- %s seconds ---" % (time.time() - start_time))



