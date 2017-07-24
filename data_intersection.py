import pandas as pd
import numpy as np
def scoring_data_intersection(df_no_pca, scoring_df):
        deindexing=scoring_df
        deindexing.reset_index(['CUSTOMER_KEY'], inplace=True)
        # deindexing.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
        new_list2=list(set(list(df_no_pca)) & set(list(deindexing)))
        df_scoring2=df_no_pca[new_list2]

        print "intersection of dataframes is done"

        return df_scoring2, new_list2