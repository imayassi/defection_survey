from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.svm import SVR, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, mutual_info_regression,VarianceThreshold
from sklearn.linear_model import RandomizedLasso, RandomizedLogisticRegression
from sklearn import cluster


depth_panelty=0.0005
def feature_clustering(df_no_pca,response, i):

    # y=bool_df[response] #use this funtion if abandoned was the response
    Y = df_no_pca[response]
    x=df_no_pca.drop([response], axis=1)
    # randomized_lasso = RandomizedLasso(selection_threshold=0.05,normalize=False,random_state= np.random.RandomState(0) ).fit_transform(x,Y)
    # new_x = randomized_lasso

    # randomized_logistic = RandomizedLogisticRegression(selection_threshold=0.05,normalize=False,random_state= np.random.RandomState(0) ).fit_transform(x,Y)
    # new_x = randomized_logistic
    # print new_x.shape
    # sel = VarianceThreshold(threshold=(.001 * (1 - .001)))
    # new_x = sel.fit_transform(x)
    #
    #
    # print 'new x ', new_x
    #
    # reduced_df = pd.DataFrame(new_x)
    # sig_features = list(set(reduced_df.idxmax(axis=1).values))
    # print  'sig features ', sig_features
    # x = df_no_pca
    # df_final = x[sig_features]
    # print list(df_final)
    # pca_df = reduced_df[sig_features]
    # bool = pd.DataFrame(Y, columns=[response])
    # # df_final.reset_index(level=['CUSTOMER_KEY'], inplace=True)
    # df = pd.concat([df_final, bool[response]], axis=1)
    #
    df2=pd.concat([x,Y], axis=1)


    df=df2.sample(frac=1)
    print 'sample dataframe shape is: ', df.shape
    y = df[response]
    x = df.drop([response], axis=1)

    # index=df_no_pca['CUSTOMER_KEY']
    dict = {}
    for m in range(10, 50, 10):
        # plsca = PLSRegression(n_components=m)
        # plsca.fit(x, y)
        # score = cross_val_score(plsca, x, y, scoring='r2')
        pca = PCA(n_components=m, random_state=np.random.RandomState(0))
        pca.fit_transform(x)
        x=pd.DataFrame(x)
        # svr=DecisionTreeRegressor(max_depth=1000)
        svr = DecisionTreeClassifier(max_depth=100)
        svr.fit(x,y)
        score = cross_val_score(svr, x, y, scoring='precision')
        avg = np.mean(score) * 100
        dict[m] = avg - (m * depth_panelty)
    print dict
    j = max(dict.iterkeys(), key=lambda k: dict[k])
    print 'the best # of components is:', j
    fc=i
    if fc == 'pls':
        # PLS appraoch
        print 'PLS appraoch'
        plsca = PLSRegression(n_components=j)
        plsca.fit(x, y)
        x3 = plsca.transform(x)
        string = "pls_"
        pls_column_name = [string + `i` for i in range(x3.shape[1])]
        df1 = pd.DataFrame(x3, columns=pls_column_name)
        plsca_df = pd.DataFrame(plsca.x_weights_)
        plsca_trans = plsca_df.transpose()
        # x.reset_index(['CUSTOMER_KEY'], inplace=True)
        # x.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
        reduced_df = pd.DataFrame(plsca_trans.values, columns=x.columns, index=pls_column_name)
        sig_features = list(set(reduced_df.idxmax(axis=1).values))

        x = df_no_pca
        df_final = x[sig_features]
        bool = pd.DataFrame(Y, columns=[response])
        # bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)

        df = pd.concat([df_final, bool[response]], axis=1)
        # df.set_index('CUSTOMER_KEY', inplace=True)

    #     PCA Approach
    elif fc == 'pca':
        pca = PCA(n_components=j, random_state=np.random.RandomState(0))
        pca.fit(x)
        x3 = pca.transform(x)
        string = "pca_"
        pca_column_name = [string + `i` for i in range(x3.shape[1])]
        reduced_df = pd.DataFrame(pca.components_, columns=x.columns, index=pca_column_name)
        sig_features = list(set(reduced_df.idxmax(axis=1).values))

        x = df_no_pca
        df_final = x[sig_features]

        pca_df = reduced_df[sig_features]
        bool = pd.DataFrame(Y, columns=[response])
        # df_final.reset_index(level=['CUSTOMER_KEY'], inplace=True)
        df = pd.concat([df_final, bool[response]], axis=1)

        # df.set_index('CUSTOMER_KEY', inplace=True)

        plsca=pca


    elif fc == 'fa':
        x = df_no_pca.drop([response], axis=1)
        agglo = cluster.FeatureAgglomeration(n_clusters=j)
        x3 = agglo.fit_transform(x)
        string = "fa_"
        df_final = pd.DataFrame(x3)
        bool = pd.DataFrame(Y, columns=[response])
        df = pd.concat([df_final, bool[response]], axis=1)
        labels=pd.DataFrame(agglo.labels_,columns=['Feature_cluster'], index=list(x)).sort_values(['Feature_cluster'], inplace=True)
        print labels
        plsca = agglo

    else:

        df=df_no_pca
        plsca = []

    return df, df[response], plsca