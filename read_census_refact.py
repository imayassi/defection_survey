import pandas as pd
from os.path import dirname, join
import numpy as np


def impute_missing_percentages(census_df=None, priors=None):
    """Imputes the missing percentages for the census data.
    To impute the values, it replaces the missing values as (missing prob)*weight_i/sum(weights of missing).
    Parameters:
        census_df(pd.DataFrame): The census data
        priors(dict): a dict with the priors
    Returns:
        census_df(pd.DataFrame): the data frame with the imputed values"""

    if census_df is None:
        census_df = pd.read_csv( 'census.csv', sep=',', header=0,  engine='python',
                              keep_default_na=True,na_values='(S)',
                              usecols=['pctwhite','pctblack',  'pctapi',  'pctaian',  'pct2prace',  'pcthispanic'  ])
        name_df = pd.read_csv('census.csv', sep=',', header=0, engine='python',
                                keep_default_na=True, na_values='(S)',
                                usecols=['name'])
        census_df.columns = ['White',  'Black', 'Asian', 'Native American','Mixed', 'Latino']
        census_df = census_df.astype(float)/100

    if priors is None:
        priors= {'White':.70,  'Black':.10, 'Asian':.05, 'Native American':.05,'Mixed':0.05, 'Latino':0.05}

    # create a copy of the data that replaces not null values as zeros
    not_null_as_zeros_df =census_df.copy()
    not_null_as_zeros_df[pd.isnull(census_df)==False]=0

    # fill na values of the data with zero
    census_df.fillna(0,inplace = True)

    # fill in the na values of the not_null_as_zeros_df using the remaining percentage * weighted priors
    # the code adjust the weights
    not_null_as_zeros_df.fillna(priors,inplace = True)
    inverse_row_sums = 1/not_null_as_zeros_df.sum(axis = 1)
    inverse_row_sums[inverse_row_sums==np.inf]=0
    not_null_as_zeros_df=not_null_as_zeros_df.multiply(inverse_row_sums,axis = 0)
    not_null_as_zeros_df=not_null_as_zeros_df.multiply(1-census_df.sum(axis = 1),axis =0).apply(np.abs)
    name_df['isLatino']=(census_df+not_null_as_zeros_df)['Latino'].apply(lambda x: True if x>= 0.5 else False)
    name_df['name']=name_df['name'].astype(str)



    return pd.concat([ (census_df+not_null_as_zeros_df), name_df], axis=1)
    # return name_df

print impute_missing_percentages(census_df=None, priors=None)


def _get_ngram_(name, i=0, n=3, allow_padding = False):
    """helper function that returns the ith ngram of size n
    Parameters:
        name (str):  The name
        i(int): index
        n(int): ngram size
        allow_padding(bool): If True, return partial emtpy
    Returns:
        ngram(str)"""

    # set up the empty string which is a boolean of size n (in class we will set this up as class variable)
    empty = ' '*n

    # get the length of the name
    l= len(name)

    # if the ith ngram is internal, return it.
    if i+n<l+1:
        return name[i: i+n]

    # if the ngram is not fully contained in the string,
    # if allow_padding =False return empty, else return a modified string.
    if allow_padding:
        return name[i:].ljust(n)
    return empty

def _count_ith_ngrams_(names, i=0, n=3, allow_padding = False):
    """counts the occurance of the ngram between staring at index i of size n.
    Parameters:
        names (pd.series): the names
        i (int): the index
        n (int): gthe ngram size
        """
    # get the ith ngram
    ngrams = names.apply(lambda x: _get_ngram_(x,i,n,allow_padding))

    # use value counts to return the counts of the ngram
    return ngrams.value_counts()


def _get_ngram_dataframe_(df, i=0, n=3, allow_padding=False):
    """creates a data frame with for the ith ngram
    Parameters:
        df (pd.dataframe) the names data frame
        i(int): The index
        n(int): ngram size
        allow_padding(bool): If true use padding"""

    # get the latino names
    latino = df.query('isLatino==True')
    non_latino = df.query('isLatino ==False')

    # Compute the probability of the four columns as a series


    last_name_latino = _count_ith_ngrams_(latino.name, i, n, allow_padding)

    last_name_non_latino = _count_ith_ngrams_(non_latino.name, i, n, allow_padding)

    # concat the series into a data frame
    dfs = pd.concat([last_name_latino, last_name_non_latino], axis=1)

    # convert nas to zero
    dfs.fillna(0, inplace=True)

    # convert the reset the indices
    dfs.reset_index(inplace=True)

    # rename the columns
    dfs.columns = ['n-gram',  'LatinoLastName', 'NonLatinoLastName']

    # return the dfs
    return dfs

print _get_ngram_dataframe_(impute_missing_percentages(census_df=None, priors=None),i=0, n=3, allow_padding = False)