import pandas as pd
import numpy as np
def get_arrays(dummy_pca,response, i, j):
    do_pca=i
    b_pca=j

    # if do_pca=='True' and b_pca=='True':
    Y = dummy_pca[response]
    X = dummy_pca.drop(response, 1)
    y = Y
    x = X
    #
    # elif do_pca=='True' and b_pca!='True':
    #
    #     Y = df_pca[response]
    #     X = df_pca.drop(response, 1)
    #     y = Y
    #     x = X
    #
    # else:
    #     Y = bool_df[response]
    #     X = df_no_pca.drop([response], axis=1)
    #     y = Y
    #     x = X

    return x, y
