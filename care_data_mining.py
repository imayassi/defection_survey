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
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

import pyodbc
import pandas as pd
random_state = np.random.RandomState(0)


conn = pyodbc.connect(dsn='VerticaProd')
random_state = np.random.RandomState(0)
cont_features=['ABANDONED','isautoclosed',	'total_agent_interaction_seconds','axc_care',	'call_smartlook_metric',	'wrapup_code_cnt',	'handled_cnt',	'ivr_decline_cnt',	'ivr_failure_cnt',	'entering_contact_us_experience',	'web_help',	'visit_num']
bool_features=['ABANDONED','axc_care',	'call_smartlook_metric',	'entering_contact_us_experience',	'web_help']
catag_features=['call_segments',	'segment_key',	'video_flag',	'care_referrer',	'first_hs_intent',	'first_contact_channel',	'ans_cr_val',	'ans_exp_val',	'ans_np_val',	'ans_tt_cares_val',	'ans_conf_corr_val',	'ans_easy_resolve_val',	'cobrowse_flag',	'source_form_desc',	'first_search_manual_location_detail',	'skill_id',	'channel',	'srs_category',	'workgroup',	'wait_time',	'case_creator_type']


data = "SELECT * FROM (SELECT ABANDONED,axc_care,	call_smartlook_metric,	wrapup_code_cnt,	handled_cnt,	ivr_decline_cnt,	ivr_failure_cnt,	entering_contact_us_experience,	web_help,	visit_num,	call_segments,	segment_key,	isautoclosed,	total_agent_interaction_seconds,	video_flag,	care_referrer,	first_hs_intent,	first_contact_channel,	ans_cr_val,	ans_exp_val,	ans_np_val,	ans_tt_cares_val,	ans_conf_corr_val,	ans_easy_resolve_val,	cobrowse_flag,	source_form_desc,	first_search_manual_location_detail,	skill_id,	channel,	srs_category,	workgroup,	wait_time,	case_creator_type FROM  CTG_ANALYTICS_WS.SM_CARE_DATA_MINING  )A   order by random() limit 1000"
df = pd.read_sql(data, conn,  coerce_float=False)

df_cont = df[cont_features]
df_cont.columns = df_cont.columns.str.strip()
df_cont.fillna(value=0, inplace=True)
df_cont.replace(to_replace=('(null)', 'NA', 'None'), value=0)
df_bool = df_cont[bool_features]
df_cont.drop(bool_features, axis=1, inplace=True)


data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)



df_char = df[catag_features]
df_char.columns = df_char.columns.str.strip()
df_char.fillna(value='-1', inplace=True)
df_char.replace(to_replace=('(null)', 'NA', 'None'), value='-1')
just_dummies = pd.get_dummies(df_char)

df_trans = pd.concat([df_bool, just_dummies, data_scaled], axis=1)

print df_trans