from sklearn.cluster import KMeans, Birch, AffinityPropagation
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
dict={}
depth_panelty=0.05
def obs_clustering(df,response,clust):
    if clust=='True':
        df2 = df.sample(frac=0.1)
        print df2.shape
        y = df2[response]
        x = df2.drop([response], axis=1)


        for m in range(5, 20, 5):
            # , random_state = np.random.RandomState(0)
            kmeans = KMeans(n_clusters=m)
            x_kmeans = kmeans.fit_transform(x)
            clustered_df = pd.DataFrame(x_kmeans)
            dt = DecisionTreeClassifier(max_depth=1000)
            # dt = DecisionTreeRegressor(max_depth=1000)
            dt.fit(clustered_df, y)
            score = cross_val_score(dt, clustered_df, y, scoring='precision')
            avg_score = np.mean(score) * 100
            print avg_score
            dict[m] = avg_score
        # print dict
        j = max(dict.iterkeys(), key=lambda k: dict[k])
        print j

        x=df.drop([response], axis=1)
        KMeans(n_clusters=m)

        X = kmeans.fit_transform(x)


        cluster_labels=pd.DataFrame(kmeans.labels_, columns=['LABELS'])
        print cluster_labels['LABELS'].value_counts()
        clustered_df = pd.DataFrame(X)
        df_final=pd.concat([clustered_df,cluster_labels,df], axis=1)
        df_orig_with_clust_labels=pd.concat([cluster_labels,df], axis=1)
        df_clusters_with_labels=pd.concat([clustered_df,cluster_labels], axis=1)
        # print df_final,df_orig_with_clust_labels,df_clusters_with_labels
    else:
        df_final=df
        df_orig_with_clust_labels=[]
        df_clusters_with_labels=[]
    return df_final,df_orig_with_clust_labels,df_clusters_with_labels
