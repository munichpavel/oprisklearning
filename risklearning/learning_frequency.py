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
#%%
    # Normalize tenors to start at -1 and preserve 0
    counts_df['t_ned'] = -counts_df['t']/counts_df['t'].min()    

    # Encode level 1 and level 2 loss categories
    le = preprocessing.LabelEncoder()
    l1_codes = le.fit_transform(counts_df['OR Category L1'])
    l2_codes = le.fit_transform(counts_df['OR Category L2'])
    ls = pd.DataFrame({'l1_codes': l1_codes, 'l2_codes': l2_codes})

    enc = preprocessing.OneHotEncoder(sparse = False)
    l_codes = pd.DataFrame(enc.fit_transform(ls))
    loss_counts = pd.concat([counts_df, l_codes], axis = 1)
    
    # Prep for neural network training
    # Select nn-relevant columns and sort by current_deltas

#    cols_nn = ['counts', 't'] + list(range(l_codes.shape[1]))
    cols_nn = ['counts', 't','t_ned'] + list(range(l_codes.shape[1]))

    #loss_counts_nn = loss_counts[cols_nn].sort_values('t')
    loss_counts_nn = loss_counts[cols_nn].sort_values('t_ned')
    
    
    # Expand data frame for day - categories with no events
    l_codes_unique = l_codes.drop_duplicates()
    n_codes = l_codes_unique.shape[0]
    
    l_codes_unique.loc[:, 'index_new'] = range(n_codes)
    l_codes_unique = l_codes_unique.set_index('index_new')
 
    # Create one df block per day with all l_codes, initialized to 0
    nn_list = [add_tenor(t, l_codes_unique) for t in 
                range(int(loss_counts_nn['t'].min()), int(loss_counts_nn['t'].max()))]

    data_nn = pd.concat(nn_list, axis = 0)
    # Reindex to avoid duplicates
    data_nn['index_new'] = range(data_nn.shape[0])
    data_nn = data_nn.set_index('index_new')

    # Merge with loss_counts by tenor and level 1/2 codes
    left_cols = ['tenor'] + list(range(l_codes.shape[1]))
    right_cols = ['t'] + list(range(l_codes.shape[1]))

     
    data_nn_bins = data_nn.merge(loss_counts_nn, left_on = left_cols, right_on = right_cols, how = 'left')
    # 'current deltas' has nans wherever no loss for given category / tenor
#%%    
    data_nn_bins = data_nn_bins.drop(['t', 'tenor'],1)
    # Replace nans with 0 for counts
    data_nn_bins['counts'] = data_nn_bins['counts'].fillna(0)
#%%
    #% Perform binning
    bin_tops_np = np.array(bin_tops)
    bin_labels = list(range(len(bin_tops_np)))
    
    cts = data_nn_bins['counts']
    
    # Note bins defined in "digitize" by lower <= x < upper
    data_nn_bins['count_bin'] = np.digitize(cts, bin_tops_np)
    # Drop counts
    data_nn_bins = data_nn_bins.drop('counts',1)
   
    #% Split into training and testing
#    data_train = data_nn_bins[data_nn_bins['tenor'] < 0]
#    data_test = data_nn_bins[data_nn_bins['tenor'] >= 0]

    data_train = data_nn_bins[data_nn_bins['t_ned'] < 0]
    data_test = data_nn_bins[data_nn_bins['t_ned'] >= 0]
    
    x_train_df = data_train.drop('count_bin',1)    
    x_test_df = data_test.drop('count_bin',1)
  
    # Encode bins (i.e. outputs) with one-hot-encoding
    enc = preprocessing.OneHotEncoder(sparse = False)
    y_train_df = bins2vecs(data_train['count_bin'], bin_labels, enc)
    #y_train_df = data_train['count_bin'] # No encoding, keep metric on bins
    y_test_df  = bins2vecs(data_test['count_bin'], bin_labels, enc)
    #y_test_df = data_test['count_bin'] # No encoding, keep metric on bins
    #%% Convert to numpy arrays for keras / tensorflow
    x_train = x_train_df.as_matrix()
    y_train = y_train_df.as_matrix()
    x_test = x_test_df.as_matrix()
    y_test = y_test_df.as_matrix()
#%%    
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
    
#%%
def bin_probs(rv, bin_tops):
    """
    Calculates pdf of random variable w.r.t. bin_tops
    
    Note that binning (a la numpy.digitize) follows convention bottom <= x < top
    """
    count_tops = [top - 1 for top in bin_tops]
    prob_tops = rv.cdf(count_tops)
    prob_left_shift = rv.cdf(count_tops[1:])
    prob_bins = np.insert(prob_left_shift - prob_tops[:-1], 0, prob_tops[0])
    return(prob_bins)

    
