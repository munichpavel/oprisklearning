# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:16:49 2016

@author: pavel
"""

%cd /home/pavel/Code/Python/risklearning

#%%
import pandas as pd
import numpy as np

from sklearn import preprocessing

#%%
# Read in loss data (counts per day / loss category)

days_year = 365
loss_ct_file = 'data/event_counts.csv'
loss_counts_raw = pd.read_csv(loss_ct_file)
#%%
# Encode level 1 and level 2 loss categories
le = preprocessing.LabelEncoder()
l1_codes = le.fit_transform(loss_counts_raw['OR Category L1'])
l2_codes = le.fit_transform(loss_counts_raw['OR Category L2'])
ls = pd.DataFrame({'l1_codes': l1_codes, 'l2_codes': l2_codes})

#%
enc = preprocessing.OneHotEncoder(sparse = False)
l_codes = pd.DataFrame(enc.fit_transform(ls))

loss_counts = pd.concat([loss_counts_raw, l_codes], axis = 1)
#%%
# Prep for neural network training
cols_nn = ['counts', 'current_delta'] + list(range(l_codes.shape[1]))
# Select nn-relevant columns and sort by current_deltas
loss_counts_nn = loss_counts[cols_nn].sort_values('current_delta')

# Shrink for testing TODO Fixme!!!
t_start = -1*days_year
t_end = days_year

loss_counts_nn = loss_counts_nn[(loss_counts_nn['current_delta'] >= t_start)
                                & (loss_counts_nn['current_delta'] < t_end)]


#%% Bin count data

# First create df with 
n_data_tenors = loss_counts_nn['current_delta'].max() - loss_counts_nn['current_delta'].min() + 1
#%%
l_codes_unique = l_codes.drop_duplicates()
n_codes = l_codes_unique.shape[0]
l_codes_unique['index_new'] = range(n_codes)
l_codes_unique = l_codes_unique.set_index('index_new')
#%%

def add_tenor(tenor, df):
    """
    Warning: DFs must have the same indices: TODO fix this
    """
    n_rows = df.shape[0]
    tenor_df = pd.DataFrame({'tenor':np.repeat(tenor, n_rows)})
    return(pd.concat([df, tenor_df], axis = 1))

#%%
# Create one df block per day with all l_codes
nn_list = [add_tenor(t, l_codes_unique) for t in 
    range(loss_counts_nn['current_delta'].min(), loss_counts_nn['current_delta'].max())]
    
data_nn = pd.concat(nn_list, axis = 0)
# Reindex to avoid duplicates
data_nn['index_new'] = range(data_nn.shape[0])
data_nn = data_nn.set_index('index_new')
#%%
# Bin data

# Merge with loss_counts by tenor and level 1/2 codes
#data_bins = pd.concat([data_nn, loss_counts_nn], axis = 1)
left_cols = ['tenor'] + list(range(30))
right_cols = ['current_delta'] + list(range(30))
#% 
data_nn_bins = data_nn.merge(loss_counts_nn, left_on = left_cols, right_on = right_cols, how = 'left')
# 'current deltas' has nans wherever no loss for given category / tenor
data_nn_bins = data_nn_bins.drop('current_delta',1)
# Replace nans with 0 for counts
data_nn_bins['counts'] = data_nn_bins['counts'].fillna(0)
#%% Perform binning
cts_max = data_nn_bins['counts'].max()
#%%
bin_tops = np.array([1,5,10,15])
bin_inds = pd.DataFrame({'bin_index': range(len(bin_tops))})
#%%
cts = data_nn_bins['counts']
data_nn_bins['count_bin'] = np.digitize(cts, bin_tops)
# Drop counts
data_nn_bins = data_nn_bins.drop('counts',1)

#%% Split into training and testing
data_train = data_nn_bins[data_nn_bins['tenor'] < 0]
data_test = data_nn_bins[data_nn_bins['tenor'] >= 0]
x_train = data_nn_bins.drop('count_bin',1)
y_train_int = data_train['count_bin']
#%% Encode bins
# Tranform bins to one-hot-vectors
enc = preprocessing.OneHotEncoder(sparse = False)
bin_vals = list(range(len(bin_tops)))
#%%
enc.fit_transform(pd.DataFrame(bin_vals))
#%
#%
y_train_list = [pd.DataFrame(enc.transform(b)) for b in y_train_int]
#%%
y_train_vecs = pd.concat(y_train_list, ignore_index = True) 
#%%
def bins2vecs(bin_df, bin_vals, enc):
    enc.fit_transform(pd.DataFrame(bin_vals))
    bin_list = [pd.DataFrame(enc.transform(b)) for b in bin_df]
    bin_vecs = pd.concat(bin_list, ignore_index = True)
    return(bin_vecs)
#%%
enc = preprocessing.OneHotEncoder(sparse = False)
y_train = bins2vecs(data_train['count_bin'], bin_vals, enc)
y_test  = bins2vecs(data_test['count_bin'], bin_vals, enc)
