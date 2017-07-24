import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, NMF
from binning import bin

def bin_pca(df_no_pca,response, j):
    df_pca=df_no_pca
    b_pca=j
    if b_pca=='True':
        pca_leng = {}
        y=df_pca[response]
        df_pca.drop([response], axis=1, inplace=True)
        lists=list(df_pca.select_dtypes(exclude=[np.bool]))
        df_cont=df_pca.select_dtypes(exclude=[np.bool])

        for k in lists:
            print k
            df_bin =pd.concat([y,df_pca[k]], axis=1) # when abandoned is the response use this function
            dict = bin(df_bin, response)
            leng = len(dict)
            # if leng>2:

            dict_list = [dict[i] for i in dict]
            for m in dict_list:

                df_pca[k+m[0]+repr(m[1])] = 0
                if m[0]=='<=' and len(m)==2:
                    df_pca[k+m[0]+repr(m[1])][df_pca[k]<=m[1]]=1

                elif m[0]=='>=' and len(m) ==2:
                    df_pca[k+m[0] + repr(m[1])][df_pca[k] >= m[1]] = 1

                elif len(m)==4:
                    df_pca[k+m[0] + repr(m[1]) + m[2] + repr(m[3])] = 0
                    df_pca[k+m[0] + repr(m[1])+m[2]+repr(m[3])][(df_pca[k]>m[1])&(df_pca[k]<m[3])] = 1
            df_pca.drop([k], axis=1, inplace=True)


        print list(df_pca)

        df = pd.get_dummies(df_pca)
            # .astype('bool')
        training_df=pd.concat([df, y], axis=1)
    else:
        training_df=df_pca
        pca_leng={}
    return   training_df, pca_leng

# dict={1:'a', 2:'b'}
# leng = len(dict)
# dict_list = [dict[i] for i in dict]
# print dict_list