#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import argparse
import sys
import logging

import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn import preprocessing

import math


from risklearning import __version__

__author__ = "munichpavel"
__copyright__ = "munichpavel"
__license__ = "none"

_logger = logging.getLogger(__name__)

def prep_count_data(counts_df, bin_tops):
    """
    Prepares loss count data for neural network training

    Args:
        counts_df (pandas df): counts of events per day / category
        bin_tops (list): upper bounds for binning
        
        Recall numpy.digitize defines bins by bottom <= x < top
        
    Returns:
        List of 4 numpy arrays: training data (input / output) and testing data (input / output)
        
        Note that training and testing are split via negative tenor (training) 
        and non-negative tenor (testing)
    """

    # Encode level 1 and level 2 loss categories
    le = preprocessing.LabelEncoder()
    l1_codes = le.fit_transform(counts_df['OR Category L1'])
    l2_codes = le.fit_transform(counts_df['OR Category L2'])
    ls = pd.DataFrame({'l1_codes': l1_codes, 'l2_codes': l2_codes})

    enc = preprocessing.OneHotEncoder(sparse = False)
    l_codes = pd.DataFrame(enc.fit_transform(ls))
    loss_counts = pd.concat([counts_df, l_codes], axis = 1)
    
    # Prep for neural network training
    cols_nn = ['counts', 'current_delta'] + list(range(l_codes.shape[1]))
    # Select nn-relevant columns and sort by current_deltas
    loss_counts_nn = loss_counts[cols_nn].sort_values('current_delta')
    
    #% Bin count data
    
    # First create df with 
    #n_data_tenors = loss_counts_nn['current_delta'].max() - loss_counts_nn['current_delta'].min() + 1
    #
    l_codes_unique = l_codes.drop_duplicates()
    n_codes = l_codes_unique.shape[0]
    
    l_codes_unique.loc[:, 'index_new'] = range(n_codes)
    l_codes_unique = l_codes_unique.set_index('index_new')
    
    # Create one df block per day with all l_codes
    nn_list = [add_tenor(t, l_codes_unique) for t in 
        range(int(loss_counts_nn['current_delta'].min()), int(loss_counts_nn['current_delta'].max()))]

    data_nn = pd.concat(nn_list, axis = 0)
    # Reindex to avoid duplicates
    data_nn['index_new'] = range(data_nn.shape[0])
    data_nn = data_nn.set_index('index_new')

    # Bin data
    
    # Merge with loss_counts by tenor and level 1/2 codes
    #data_bins = pd.concat([data_nn, loss_counts_nn], axis = 1)
    left_cols = ['tenor'] + list(range(l_codes.shape[1]))
    right_cols = ['current_delta'] + list(range(l_codes.shape[1]))
     
    data_nn_bins = data_nn.merge(loss_counts_nn, left_on = left_cols, right_on = right_cols, how = 'left')
    # 'current deltas' has nans wherever no loss for given category / tenor
    data_nn_bins = data_nn_bins.drop('current_delta',1)
    # Replace nans with 0 for counts
    data_nn_bins['counts'] = data_nn_bins['counts'].fillna(0)

    #% Perform binning
   # cts_max = data_nn_bins['counts'].max()
    
    bin_tops_np = np.array(bin_tops)
    bin_labels = list(range(len(bin_tops_np)))
    
    cts = data_nn_bins['counts']
    
    # Note bins defined in "digitize" by lower <= x < upper
    data_nn_bins['count_bin'] = np.digitize(cts, bin_tops_np)
    # Drop counts
    data_nn_bins = data_nn_bins.drop('counts',1)
   
    #% Split into training and testing
    data_train = data_nn_bins[data_nn_bins['tenor'] < 0]
    data_test = data_nn_bins[data_nn_bins['tenor'] >= 0]
    
    x_train = data_train.drop('count_bin',1)
    #y_train_int = data_train['count_bin']

    x_train_df = data_train.drop('count_bin',1)    
    # Encode bins with one-hot-encoding
    #TODO check on error with simulated loss counts
    enc = preprocessing.OneHotEncoder(sparse = False)
    y_train_df = bins2vecs(data_train['count_bin'], bin_labels, enc)
    #y_train_df = data_train['count_bin'] # No encoding, keep metric on bins
    x_test_df = data_test.drop('count_bin',1)
    y_test_df  = bins2vecs(data_test['count_bin'], bin_labels, enc)
    #y_test_df = data_test['count_bin'] # No encoding, keep metric on bins
    #%% Convert to numpy arrays for keras / tensorflow
    x_train = x_train_df.as_matrix()
    y_train = y_train_df.as_matrix()
    x_test = x_test_df.as_matrix()
    y_test = y_test_df.as_matrix()
    
    return([x_train, y_train, x_test, y_test])

#%%
def bins2vecs(bin_df, bin_labels, enc):
        enc.fit_transform(pd.DataFrame(bin_labels))
        bin_list = [pd.DataFrame(enc.transform(b)) for b in bin_df]
        bin_vecs = pd.concat(bin_list, ignore_index = True)
        return(bin_vecs)
    
def add_tenor(tenor, df):
    """
    Warning: DFs must have the same indices: TODO fix this
    """
    n_rows = df.shape[0]
    tenor_df = pd.DataFrame({'tenor':np.repeat(tenor, n_rows)})
    return(pd.concat([df, tenor_df], axis = 1))
#%%
def sim_counts(freq_param_init, freq_param_final, n_tenors):
    """
    Simulate Poisson process with linearly changing intensity parameter
    """
    freq_rv = stats.poisson
    lambda_ts = np.linspace(freq_param_init, freq_param_final, num=n_tenors)
    counts = [freq_rv(l_t).rvs() for l_t in lambda_ts]
    return(counts)
    
