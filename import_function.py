import pyodbc

import pandas as pd
from sklearn import preprocessing
import numpy as np
conn = pyodbc.connect(dsn='VerticaProd')

def import_scoring_data(scoring_data,scoring_data_PY, scoring_data_PY2, cont_score_features, bool_score_features, catag_score_features):
    df = pd.read_sql(scoring_data, conn, index_col='AUTH_ID', coerce_float=False)
    df3 = pd.read_sql(scoring_data_PY, conn, index_col='AUTH_ID', coerce_float=False)
    df4 = pd.read_sql(scoring_data_PY2, conn, index_col='AUTH_ID', coerce_float=False)

    df_cont = df[cont_score_features]
    df_cont.columns = df_cont.columns.str.strip()
    df_cont.fillna(value=0, inplace=True)
    avg=df_cont.mean(axis=0, skipna=True)
    a=list(avg)
    for i in range(len(a)):
        for j in df_cont:
            df_cont[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont = df_cont.astype(float)
    df_bool = df_cont[bool_score_features]
    for f in df_bool.columns:
        if len(df_bool[f].unique()) < 2:
            df_bool.drop([f], axis=1, inplace=True)

    bool = df_bool.astype('bool')
    df_bool = bool
    df_cont.drop(bool_score_features, axis=1, inplace=True)

    index_df = pd.DataFrame(df_cont.reset_index(level=['AUTH_ID']), columns=['AUTH_ID'])

    data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
    data_scaled = pd.concat([data_scaled, index_df], axis=1)
    data_scaled.set_index('AUTH_ID', inplace=True)

    df_char = df[catag_score_features]
    df_char = df_char.astype(object)

    df_char.columns = df_char.columns.str.strip()
    df_char.fillna(value='-1', inplace=True)
    df_char.replace(to_replace=('(null)', 'NA', 'None', '', ' ', '\t'), value='-1')
    just_dummies = pd.get_dummies(df_char).astype('bool')
    # **************************************************************
    # ***************************importing base PY df***************
    # **************************************************************
    df_cont_py = df3[cont_score_features]
    df_cont_py.columns = df_cont_py.columns.str.strip()
    df_cont_py.fillna(value=0, inplace=True)
    med = df_cont_py.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_py:
            df_cont_py[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont_py = df_cont_py.astype(float)
    df_bool_py = df_cont_py[bool_score_features]
    bool = df_bool_py.astype('bool')

    df_bool_py = bool

    df_cont_py.drop(bool_score_features, axis=1, inplace=True)
    print 'df_cont_py done'
    df_char_py = df3[catag_score_features]
    df_char_py.columns = df_char_py.columns.str.strip()
    df_char_py.fillna(value='-1', inplace=True)
    df_char_py.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies_py = pd.get_dummies(df_char_py).astype('bool')
    print 'just_dummies_py done'
    print 'df_trans_py done'

    # **************************************************************
    # ***************************importing base PY2 df***************
    # **************************************************************

    df_cont_py2 = df4[cont_score_features]
    df_cont_py2.columns = df_cont_py2.columns.str.strip()
    df_cont_py2.fillna(value=0, inplace=True)
    med = df_cont_py2.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_py2:
            df_cont_py2[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont_py2 = df_cont_py2.astype(float)
    df_bool_py2 = df_cont_py2[bool_score_features]
    bool = df_bool_py2.astype('bool')
    df_bool_py2 = bool

    df_cont_py2.drop(bool_score_features, axis=1, inplace=True)
    print 'df_cont_py2 done'
    df_char_py2 = df4[catag_score_features]
    df_char_py2.columns = df_char_py2.columns.str.strip()
    df_char_py2.fillna(value='-1', inplace=True)
    df_char_py2.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies_py2 = pd.get_dummies(df_char_py2).astype('bool')
    print 'just_dummies_py2 done'
    df_trans_py2 = pd.concat([df_bool_py2, just_dummies_py2, df_cont_py2], axis=1)
    print 'df_trans_py2 done'

    # **************************************************************
    # ******joining df_trans & df_trans_py & df_trans_py2**********
    # **************************************************************
    boolean_df = pd.concat([df_bool, just_dummies], axis=1)
    boolean_df_py = pd.concat([df_bool_py, just_dummies_py], axis=1)
    boolean_df_py2 = pd.concat([df_bool_py2, just_dummies_py2], axis=1)
    df_j1 = boolean_df.join(boolean_df_py, rsuffix='_py').astype('bool').fillna(value='False')
    df_j2 = df_cont.join(df_cont_py, rsuffix='_py').astype(float).fillna(value=0)
    df_j3 = df_j1.join(boolean_df_py2, rsuffix='_py2').astype('bool').fillna(value='False')
    df_j4 = df_j2.join(df_cont_py2, rsuffix='_py2').astype(float).fillna(value=0)
    df_trans = pd.concat([df_j3, df_j4], axis=1)
    df_trans.fillna(value='False')

    for column in df_trans.columns:
        if df_trans[column].dtype == np.bool:
            df_trans[column] = df_trans[column].fillna(value='False')
        else:
            df_trans[column] = df_trans[column].fillna(df_trans[column].median())

    # **************************************************************
    # ***************************importing care df******************
    # **************************************************************

    full_df = pd.concat([df_j3, df_j4], axis=1)
    print full_df
    return full_df


def import_data(data,data_PY,data_PY2, cont_features, bool_features,response, catag_features):
    df = pd.read_sql(data, conn,index_col='AUTH_ID', coerce_float=False)
    df3 = pd.read_sql(data_PY, conn, index_col='AUTH_ID', coerce_float=False)
    df4 = pd.read_sql(data_PY2, conn, index_col='AUTH_ID', coerce_float=False)
    df[df['TAX_YEAR']==2015]
    df3[df3['TAX_YEAR'] == 2014]
    df4[df4['TAX_YEAR'] == 2013]
    df.drop(['TAX_YEAR'], axis=1, inplace=True)
    df3.drop(['TAX_YEAR'], axis=1, inplace=True)
    df4.drop(['TAX_YEAR'], axis=1, inplace=True)
                    # **************************************************************
                    # ***************************importing base data df*************
                    # **************************************************************
    df_cont = df[cont_features]
    df_cont.columns = df_cont.columns.str.strip()
    df_cont.fillna(value=0, inplace=True)
    med=df_cont.median(axis=0, skipna=True)
    a=list(med)
    for i in range(len(a)):
        for j in df_cont:
            df_cont[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont = df_cont.astype(float)
    df_bool = df_cont[bool_features]
    bool=df_bool.astype('bool')
    df_bool=bool

    df_cont.drop(bool_features, axis=1, inplace=True)
    print 'df_cont done'
    df_char = df[catag_features]
    df_char.columns = df_char.columns.str.strip()
    df_char.fillna(value='-1', inplace=True)
    df_char.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies = pd.get_dummies(df_char).astype('bool')
    print 'just_dummies done'
    print 'df_trans done'
    # **************************************************************
    # ***************************importing base PY df***************
    # **************************************************************
    df_cont_py = df3[cont_features]
    df_cont_py.columns = df_cont_py.columns.str.strip()
    df_cont_py.fillna(value=0, inplace=True)
    med = df_cont_py.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_py:
            df_cont_py[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont_py = df_cont_py.astype(float)
    df_bool_py = df_cont_py[bool_features]
    bool = df_bool_py.astype('bool')
    df_bool_py = bool

    df_cont_py.drop(bool_features, axis=1, inplace=True)
    print 'df_cont_py done'
    df_char_py = df3[catag_features]
    df_char_py.columns = df_char_py.columns.str.strip()
    df_char_py.fillna(value='-1', inplace=True)
    df_char_py.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies_py = pd.get_dummies(df_char_py).astype('bool')
    print 'just_dummies_py done'

    print 'df_trans_py done'


    # **************************************************************
    # ***************************importing base PY2 df***************
    # **************************************************************

    df_cont_py2 = df4[cont_features]
    df_cont_py2.columns = df_cont_py2.columns.str.strip()
    df_cont_py2.fillna(value=0, inplace=True)
    med = df_cont_py2.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_py2:
            df_cont_py2[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont_py2 = df_cont_py2.astype(float)
    df_bool_py2 = df_cont_py2[bool_features]
    bool = df_bool_py2.astype('bool')
    df_bool_py2 = bool

    df_cont_py2.drop(bool_features, axis=1, inplace=True)
    print 'df_cont_py2 done'
    df_char_py2 = df4[catag_features]
    df_char_py2.columns = df_char_py2.columns.str.strip()
    df_char_py2.fillna(value='-1', inplace=True)
    df_char_py2.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies_py2 = pd.get_dummies(df_char_py2).astype('bool')
    print 'just_dummies_py2 done'
    print 'df_trans_py2 done'



    # **************************************************************
    # ******joining df_trans & df_trans_py & df_trans_py2**********
    # **************************************************************
    boolean_df=pd.concat([df_bool, just_dummies], axis=1)
    boolean_df_py = pd.concat([df_bool_py, just_dummies_py], axis=1)
    boolean_df_py2 = pd.concat([df_bool_py2, just_dummies_py2], axis=1)
    df_j1=boolean_df.join(boolean_df_py,rsuffix='_py').astype('bool').fillna(value='False')
    df_j2=df_cont.join(df_cont_py, rsuffix='_py').astype(float).fillna(value=0)
    df_j3 = df_j1.join(boolean_df_py2, rsuffix='_py2').astype('bool').fillna(value='False')
    df_j4 = df_j2.join(df_cont_py2, rsuffix='_py2').astype(float).fillna(value=0)
    df_trans=pd.concat([df_j3,df_j4], axis=1)
    df_trans.fillna(value='False')

    for column in df_trans.columns:
        if df_trans[column].dtype == np.bool:
            df_trans[column] = df_trans[column].fillna(value='False')
        else:
            df_trans[column] = df_trans[column].fillna(df_trans[column].median())

    df_trans.drop(['FLAG_py','FLAG_py2'], axis=1, inplace=True)

    # **************************************************************
    # ***************************PRIOR YEARS DF*********************
    # **************************************************************
    # **************************************************************
    # **************************************************************
    # **************************************************************

    df = pd.read_sql(data, conn, index_col='AUTH_ID', coerce_float=False)
    df3 = pd.read_sql(data_PY, conn, index_col='AUTH_ID', coerce_float=False)
    df4 = pd.read_sql(data_PY2, conn, index_col='AUTH_ID', coerce_float=False)
    df[df['TAX_YEAR'] == 2014]
    df3[df3['TAX_YEAR'] == 2013]
    df4[df4['TAX_YEAR'] == 2012]
    df.drop(['TAX_YEAR'], axis=1, inplace=True)
    df3.drop(['TAX_YEAR'], axis=1, inplace=True)
    df4.drop(['TAX_YEAR'], axis=1, inplace=True)
    # **************************************************************
    # ***************************importing base data df*************
    # **************************************************************
    df_cont = df[cont_features]
    df_cont.columns = df_cont.columns.str.strip()
    df_cont.fillna(value=0, inplace=True)
    med = df_cont.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont:
            df_cont[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont = df_cont.astype(float)
    df_bool = df_cont[bool_features]
    bool = df_bool.astype('bool')
    df_bool = bool

    df_cont.drop(bool_features, axis=1, inplace=True)
    print 'df_cont done'
    df_char = df[catag_features]
    df_char.columns = df_char.columns.str.strip()
    df_char.fillna(value='-1', inplace=True)
    df_char.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies = pd.get_dummies(df_char).astype('bool')
    print 'just_dummies done'
    print 'df_trans done'
    # **************************************************************
    # ***************************importing base PY df***************
    # **************************************************************
    df_cont_py = df3[cont_features]
    df_cont_py.columns = df_cont_py.columns.str.strip()
    df_cont_py.fillna(value=0, inplace=True)
    med = df_cont_py.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_py:
            df_cont_py[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont_py = df_cont_py.astype(float)
    df_bool_py = df_cont_py[bool_features]
    bool = df_bool_py.astype('bool')
    df_bool_py = bool

    df_cont_py.drop(bool_features, axis=1, inplace=True)
    print 'df_cont_py done'
    df_char_py = df3[catag_features]
    df_char_py.columns = df_char_py.columns.str.strip()
    df_char_py.fillna(value='-1', inplace=True)
    df_char_py.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies_py = pd.get_dummies(df_char_py).astype('bool')
    print 'just_dummies_py done'

    print 'df_trans_py done'

    # **************************************************************
    # ***************************importing base PY2 df***************
    # **************************************************************

    df_cont_py2 = df4[cont_features]
    df_cont_py2.columns = df_cont_py2.columns.str.strip()
    df_cont_py2.fillna(value=0, inplace=True)
    med = df_cont_py2.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_py2:
            df_cont_py2[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont_py2 = df_cont_py2.astype(float)
    df_bool_py2 = df_cont_py2[bool_features]
    bool = df_bool_py2.astype('bool')
    df_bool_py2 = bool

    df_cont_py2.drop(bool_features, axis=1, inplace=True)
    print 'df_cont_py2 done'
    df_char_py2 = df4[catag_features]
    df_char_py2.columns = df_char_py2.columns.str.strip()
    df_char_py2.fillna(value='-1', inplace=True)
    df_char_py2.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies_py2 = pd.get_dummies(df_char_py2).astype('bool')
    print 'just_dummies_py2 done'
    print 'df_trans_py2 done'

    # **************************************************************
    # ******joining df_trans & df_trans_py & df_trans_py2**********
    # **************************************************************
    boolean_df = pd.concat([df_bool, just_dummies], axis=1)
    boolean_df_py = pd.concat([df_bool_py, just_dummies_py], axis=1)
    boolean_df_py2 = pd.concat([df_bool_py2, just_dummies_py2], axis=1)
    df_j1 = boolean_df.join(boolean_df_py, rsuffix='_py').astype('bool').fillna(value='False')
    df_j2 = df_cont.join(df_cont_py, rsuffix='_py').astype(float).fillna(value=0)
    df_j3_b = df_j1.join(boolean_df_py2, rsuffix='_py2').astype('bool').fillna(value='False')
    df_j4_b = df_j2.join(df_cont_py2, rsuffix='_py2').astype(float).fillna(value=0)
    df_trans = pd.concat([df_j3, df_j4], axis=1)
    df_trans.fillna(value='False')

    for column in df_trans.columns:
        if df_trans[column].dtype == np.bool:
            df_trans[column] = df_trans[column].fillna(value='False')
        else:
            df_trans[column] = df_trans[column].fillna(df_trans[column].median())

    df_trans.drop(['FLAG_py', 'FLAG_py2'], axis=1, inplace=True)









    # **************************************************************
    # ***************************importing care df******************
    # **************************************************************

    full_df=pd.concat([df_j3,df_j4], axis=1)
    full_df.reset_index(['AUTH_ID'], inplace=True)
    full_df.drop(['AUTH_ID'],axis=1, inplace=True)

    full_df_b=pd.concat([df_j3_b,df_j4_b], axis=1)
    full_df_b.reset_index(['AUTH_ID'], inplace=True)
    full_df_b.drop(['AUTH_ID'],axis=1, inplace=True)
    full_df.append(full_df_b)

    print full_df


    return full_df





