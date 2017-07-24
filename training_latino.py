# Class with Pandas
# import required modules

import numpy as np
import pandas as pd

from latino.custom_multinomial.loadnames import LoadNames
from latino.utils import get_ngram


class MBBase(object):
    """General Class for the training the multi-nomial Bayes for the Latino Model"""

    def __init__(self, data, n=3, s=1, max_ngrams=10, allow_padding=True, missing_method='probability'):
        """ General Class for the Multinomial Bayes.

        Parameters:
            data(pd.DataFrame): Data Frame with the Training data with columns: FirstName, LastName, isLatino
            n(int): n-gram size
            s(int): size of shift
            max_ngrams(int): Max number of n-grams
            allow_padding(bool): If True, returns padded n-grams
            missing_method(str): the method to fill in na values. Four cases:
                'probability': fill in with the minimum value for that ngram for both latino and non latino
                'zeros': fill in with zero
                'size': fill in with 1/ngram size (the size is for each column seperately).
                None: do nothing. Don't fill in.
                """

        # set attributes
        self.n = n
        self.s = s
        self.max_ngrams = max_ngrams
        self.allow_padding = allow_padding
        self.empty = ' ' * self.n
        self.missing_method = missing_method

        # set up the likelihoods
        # self.first_name_likelihoods = None
        self.last_name_likelihoods = None

        # init a dictionary of list to hold the vocab size
        self.vocab_size = {
                           'Latino Last Name': [],

                           'NonLatino Last Name': []
                           }

    def _organize_data_(self, data):
        """helper function that organizes the data"""
        # does nothing in base
        # self.first_latino = None
        self.last_latino = None
        # self.first_non_latino = None
        self.last_non_latino = None

    def __repr__(self):
        params = (self.n, self.s, self.max_ngrams, self.allow_padding, self.missing_method)
        return 'TrainMultiNomialBayes(n:%d,s:%d,max_ngrams:%d, allow_padding:%r,missing_method:%s )' % params

    def _get_ngram_(self, name, i=0):
        """helper function that returns the ith n_gram of size n, encapsulate the get_ngram function from utils
        Parameters:
            name (str):  The name
            i(int): index
        Returns:
            n_gram(str)"""

        return get_ngram(name, i=i, n=self.n, allow_padding=True)

    def _compute_ith_ngram_prob(self, names, i=0, is_latino=True):
        """counts the occurrence of the ith ngram."""
        # does nothing in base- depends on the data set
        pass

    def _get_ngram_dataframe_(self, i):
        """creates a data frame with for the ith ngram, fills in missing values and gets the vocab size
        Parameters:
            i(int): The index
        Returns:
            (first_name_df,last_name_df): the liklehoods for the ith n gram for the first and last name
        """

        # compute the probability of n-gram for first name, last name for latino and non latino # do as seperate

        # first names first
        # first_name_latino = self._compute_ith_ngram_prob(self.first_latino, i)
        # first_name_non_latino = self._compute_ith_ngram_prob(self.first_non_latino, i, False)

        # last_names_last
        last_name_latino = self._compute_ith_ngram_prob(self.last_latino, i)
        last_name_non_latino = self._compute_ith_ngram_prob(self.last_non_latino, i, False)

        # update the vocab size for each name type by appending the lengths of the values.
        # self.vocab_size['Latino First Name'].append(len(first_name_latino))
        # self.vocab_size['NonLatino First Name'].append(len(first_name_non_latino))
        self.vocab_size['Latino Last Name'].append(len(last_name_latino))
        self.vocab_size['NonLatino Last Name'].append(len(last_name_non_latino))

        # concatenate into data frames
        # first_name_df = pd.concat([first_name_latino,
        #                            first_name_non_latino],
        #                           axis=1)

        last_name_df = pd.concat([last_name_latino,
                                  last_name_non_latino],
                                 axis=1)

        # reset the indices to make the n-gram a column
        # first_name_df.reset_index(inplace=True)
        last_name_df.reset_index(inplace=True)

        # rename the columns for each.
        # first_name_df.columns = ['ngram', 'Latino', 'NonLatino']
        last_name_df.columns = ['ngram', 'Latino', 'NonLatino']

        # replace the missing_values
        # first_name_df = self._replace_missing_values_(first_name_df)
        last_name_df = self._replace_missing_values_(last_name_df)

        # return the dfs
        return  last_name_df

    def _replace_missing_values_(self, df):
        """helper function that replaces the missing values based on the desired method"""
        df = df.copy()

        # handle the cases
        if self.missing_method == 'probability':
            # fill in with the lowest probability
            min_probability = df.min().min()
            df.fillna(min_probability, inplace=True)

        if self.missing_method == 'zeros':
            # fill in with zeros
            df = df.fillna(0)

        if self.missing_method == 'size':
            # fill in by size
            sizes = df.count()  # calculate the sizes
            df.Latino = df.Latino.fillna(1.0 / sizes[0])
            df.NonLatino = df.NonLatino.fillna(1.0 / sizes[1])

        if self.missing_method is None:
            # if none do nothing just pass.
            pass
        return df

    def count_ngrams(self, return_vals=False):
        """Trains the model by calcualting the probability of the ngram_0 to ngram_max-1
        Returns:
            (list): a list of dataframes, 1 for each ngram indexed from 0 to max-1 """

        # get an array of indices
        indices = np.arange(start=0, stop=self.max_ngrams * self.s, step=self.s)

        # compute the likelihoods of the ngrams with a comprehension- this returns list of tuples
        likelihoods = [self._get_ngram_dataframe_(i) for i in indices]

        # extract the dataframes from the tuples above and put into a data frame
        # self.first_name_likelihoods = [tuple_[0] for tuple_ in likelihoods]
        self.last_name_likelihoods = [tuple_[1] for tuple_ in likelihoods]

        # return the likelihoods if desired
        if return_vals:
            return  self.last_name_likelihoods

    ### accessor functions
    def get_max_ngrams(self):
        return self.max_ngrams

    def get_n(self):
        return self.n

    def get_shift(self):
        return self.s

    def get_padding(self):
        return self.allow_padding

    # def get_first_name_likelihoods(self):
    #     return self.first_name_likelihoods

    def get_last_name_likelihoods(self):
        return self.last_name_likelihoods

    def get_vocab_size(self):
        """returns the vocab size"""
        return self.vocab_size


class TrainMultinomialBayes(MBBase):
    """General Class for the training the multi-nomial Bayes for the Latino Model"""

    def __init__(self, data, n=3, s=1, max_ngrams=10, allow_padding=False, missing_method='probability'):
        """ General Class for the Multinomial Bayes.

        Parameters:
            data(pd.DataFrame): Data Frame with the Training data with columns: FirstName, LastName, isLatino
            n(int): n-gram size
            s(int): size of shift
            max_ngrams(int): Max number of n-grams
            allow_padding(bool): If True, returns padded n-grams
            missing_method(str): the method to fill in na values. Four cases:
                'probability': fill in with the minimum value
                'zeros': fill in with zero
                'size': fill in with 1/ngram size.
                None: do nothing. Don't fill in.
             """

        MBBase.__init__(self, data, n, s, max_ngrams, allow_padding, missing_method)

        self._organize_data_(data)

    def _organize_data_(self, data, priors=None):
        """helper function that organizes the data"""

        # split the data into 4 different series and assign as attributes
        # self.first_latino = data.query('isLatino == True').FirstName
        self.last_latino = data.query('isLatino == True').LastName
        # self.first_non_latino = data.query('isLatino==False').FirstName
        self.last_non_latino = data.query('isLatino ==False').LastName

        # get the number of latino and non latinos
        # self.num_latino = len(self.first_latino) * 1.0
        # self.num_non_latino = len(self.first_non_latino) * 1.0

        # set up self.likelihoods
        self.likelihoods = None

    def __repr__(self):
        return 'TrainMultiNomialBayes(n=%d,s=%d,max_ngrams=%d)' % (self.n, self.s, self.max_ngrams)

    def _compute_ith_ngram_prob(self, names, i=0, is_latino=True):
        """ computes the probability of the ith ngram by counting.
        Parameters:
            names (pd.series): the names
            i (int): the index
            is_latino(bool): if true normalize by the latino counts, else other
        Returns:
             probs(pd.Series): the adjusted probs
            """

        # get the ith n-grams
        ngrams = names.apply(lambda x: self._get_ngram_(x, i))

        # count the n-gram
        counts = ngrams.value_counts()

        # normalize by the number of latinos or the number of non latinos and return
        norm = self.num_latino if is_latino else self.num_non_latino
        return counts / norm


class TrainMultinomialBayesRandom(MBBase):
    """General Class for the training the multi-nomial Bayes for the Latino Model"""

    def __init__(self, data, priors=None, n=3, s=1, max_ngrams=10, allow_padding=True, missing_method='probability'):
        """ Generate the training for randomNames data set.

        Parameters:
            data(pd.DataFrame):
            priors(dict): priors for the ethnicities
            n(int): n-gram size
            s(int): size of shift
            max_ngrams(int): Max number of n-grams
            allow_padding(bool): If True, returns padded n-grams
            missing_method(str): the method to fill in na values. Four cases:
                'probability': fill in with the minimum value
                'zeros': fill in with zero
                'size': fill in with 1/ngram size.
                None: do nothing. Don't fill in.
             """

        MBBase.__init__(self, data, n, s, max_ngrams, allow_padding, missing_method)
        self._organize_data_(data, priors)

    def _organize_data_(self, data, priors):
        """ helper function that organizes the data
        Parameters:
            data(pd.DataFrame): data frame with cols: NameType,Name,Ethnicity,Likelihood
            priors(dict): dict of the priors
        """

        # assign the priors
        if priors is None:
            self.priors = {'NativeAmerican': 0.01,
                           'Asian': 0.05,
                           'Black': 0.13,
                           'Latino': 0.17,
                           'White': 0.64}
        else:
            self.priors = priors

        # set up the  priors for latino and not latino
        self.prob_latino = self.priors['Latino']
        self.prob_non_latino = 1 - self.prob_latino

        # make a copy of the data to prevent contamination
        data = data.copy()

        # calculate likelihood*priors by first using the priors dict to map a prior
        # to each name based on ethnicity and then multiplying the likelihood by the mapped_priors
        priors_mapped = data.Ethnicity.map(self.priors)
        data['Prior*Likelihood'] = data['Likelihood'] * priors_mapped

        # partition the data frame into sub data frames
        # self.first_latino = data.query('Ethnicity == "Latino" & NameType =="F"')
        self.last_latino = data.query('Ethnicity == "Latino" & NameType =="L"')
        # self.first_non_latino = data.query('Ethnicity != "Latino" & NameType =="F"')
        self.last_non_latino = data.query('Ethnicity != "Latino" & NameType =="L"')

    def _compute_ith_ngram_prob(self, df, i=0, is_latino=True):
        """computes the probability of the ith ngram.
        Parameters:
            df (pd.DataFrame): DataFrame with the names info
            i (int): the index
            is_latino(bool): if true normalize by the latino prob, else other prob
        Return:
            probs (pd.Series) a series.
            """

        # get the ith n-grams
        ngrams = df.Name.apply(lambda x: self._get_ngram_(x, i))

        # group the names by the ngrams
        grouped_ngrams = df.groupby(ngrams)

        # sum the probs, which are the Priors*Likelihoods for each ngram
        probs = grouped_ngrams['Prior*Likelihood'].sum()

        # normalize by the proper normalization factor and return
        norm = self.prob_latino if is_latino else self.prob_non_latino
        return probs / norm


if __name__ == '__main__':
    data = LoadNames().load_random_set()
    trainer = TrainMultinomialBayesRandom(data=data)
    ngrams = trainer.count_ngrams(True)
    print trainer.get_vocab_size()
