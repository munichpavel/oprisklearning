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
#%%

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
    l1_codes = le.fit_transform(counts_df['L1_cat'])
    l2_codes = le.fit_transform(counts_df['L2_cat'])
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
                range(int(loss_counts_nn['t'].min()), int(loss_counts_nn['t'].max()+1))]

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
def sim_counts(freq_param_ts, freq_rv):
    """
    Simulate count process with non-stationary parameter (currently only 1d)
    """
    counts = [freq_rv(l_t).rvs() for l_t in freq_param_ts]
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

    
#%%
def rl_train_net(x_train, y_train, x_test, y_test, \
                layer_shapes, batch_size = 32, n_epoch=10, \
                optimizer='adagrad', dropout = 0.2,\
                loss_fn = 'categorical_crossentropy'):
    """
    Trains nn with given train / test data and meta-parameters via keras / TensorFlow

    Args:
        x_train, y_train, x_test, y_test (numpy): training testing data
        layer_shapes (list): defines hidden layer architecture, 
            each entry is number of neurons per layer
        batch_size, n_epoch (int): self-explanatory
        optimizer (string or keras optimizer): either standard string flag or 
            keras optimizer object
        dropout (float): dropout rate
        loss_fn (string): standard string flag for keras (for now)
        
    Returns:
        Dictionary of keras model and predictive probabilities for testing set
        
        Note that training and testing are split via negative tenor (training) 
        and non-negative tenor (testing)
    """
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
   
    # Number of nodes in output layer: if series, 1, else number of cols
    out_layer_len = 1 if len(y_train.shape)==1 else y_train.shape[1]
    model = Sequential()
    model.add(Dense(layer_shapes[0], input_shape=(x_train.shape[1],)))
    model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                              # of the layer above. Here, with a "rectified linear unit",
                              # we clamp all values below 0 to 0.
    model.add(Dropout(dropout))   # Default dropout parameter

    #% Middle layers
    for layer_size in layer_shapes:                           
        model.add(Dense(layer_size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(out_layer_len))
    model.add(Activation('softmax')) 


    model.compile(loss= loss_fn, optimizer=optimizer)
    # But with keras verbose = 1 for larger networks 
    model.fit(x_train, y_train,
          batch_size=batch_size, nb_epoch=n_epoch,
          show_accuracy=True, verbose=0,
          validation_data=(x_test, y_test))

    proba = model.predict_proba(x_test, batch_size=32)
    
    return({'model': model, 'probs_nn': proba})    
    
def probs_kl(proba, lambda_ts, t_start, t_end, bin_tops, mle_probs_vals):
    """
    Converts test output of keras from wide to long and calculated KL divergences

    Args:
        proba (numpy array): prediction output of keras / TensorFlow
        lambda_ts (mumpy array): intensity values for Poisson count process
        t_start, t_end (int): starting (train) and ending (test) tenors
        bin_tops (list): upper bounds for binning of counts        
            Recall numpy.digitize defines bins by bottom <= x < top
        mle_probs_vals (list): Poisson pdf w.r.t. bins from MLE
        
    Returns:
        Dictionary of (long) pandas df probs, 2 lists of KL divergences for 
            test data (tenors >= 0)
    """
    #% Convert proba from wide to long and append to other probs
    probs_list = []
    kl_mle_list = []
    kl_nn_list = []
    
    count_tops = [bin_top - 1 for bin_top in bin_tops]

    for t in range(proba.shape[0]):
        nn_probs_t = proba[t]    
        true_bins_t = bin_probs(stats.poisson(lambda_ts[-t_start+t]), bin_tops)
        probs_t = pd.DataFrame({'Tenor': t, 'Count Top': count_tops, \
                                'Probs True': true_bins_t, \
                                'Probs NN': nn_probs_t, \
                                'Probs MLE': mle_probs_vals}, \
                                index = range(t*len(count_tops), \
                                    t*len(count_tops) + len(count_tops)))
        probs_list.append(probs_t)
        # Calculate KL divergences
        kl_mle_list.append(stats.entropy(true_bins_t, mle_probs_vals))
        kl_nn_list.append(stats.entropy(true_bins_t, nn_probs_t))

    probs = pd.concat(probs_list)
    kl_df = pd.DataFrame({'Tenor': range(0, t_end), \
                      'KL MLE': kl_mle_list, \
                      'KL NN': kl_nn_list})
    return({'Probs': probs, 'KL df': kl_df})
