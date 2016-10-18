# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:16:49 2016

@author: pavel
"""

%cd /home/pavel/Code/Python/risklearning

#%%
import scipy.stats as stats
import numpy as np
import pandas as pd

from pandas_ply import install_ply, X, sym_call
install_ply(pd)

from sklearn import preprocessing
import math

import risklearning.learning_frequency as rlf

#%%
#%%
tenors_horizon = 365 # (Time) tenors (e.g. 1 day) per model horizon (e.g. 1 year)

h_start = 5.0 # How many model horizons of past data to train
h_end = 1.0 #How many model horizons of past data to test / validate

t_start = -int(math.floor(h_start*tenors_horizon))
t_end = int(math.floor(h_end*tenors_horizon))


#% Generate Poisson-distributed events
lambda_init = 0.5 # intensity over tenor (e.g. day)
lambda_final = 0.5 # intensity over tenor (e.g. day)
n_tenors = t_end - t_start
counts = rlf.sim_counts(lambda_init, lambda_final, n_tenors)
#%
# Build df around counts
l1s = ['Execution Delivery and Process Management']*n_tenors
l2s = ['Transaction Capture, Execution and Maintenance']*n_tenors
tenors = list(xrange(t_start, t_end))
#%
counts_sim_df = pd.DataFrame({'t': tenors,
                              'OR Category L1': l1s, 'OR Category L2': l2s,
                              'counts': counts
                             })
#%% Do MLE (simple average for Poisson process
n_tenors_train = -t_start
n_tenors_test = t_end
counts_train = (counts_sim_df.ply_where(counts_sim_df.t < 0)).groupby('OR Category L2').sum()
counts_test =  counts_sim_df.ply_where(counts_sim_df.t >= 0).groupby('OR Category L2').sum()

lambdas_train = counts_train['counts']/n_tenors_train
lambdas_test = counts_train['counts']/n_tenors_test

bin_tops = [1,6,101]
# Recall that digitize defines bins by lower <= x < upper
count_tops =[count - 1 for count in bin_tops]

# Calculate bin probabilities from MLE poisson
poi_mle = stats.poisson(lambdas_train)
poi_tops = poi_mle.cdf(count_tops)
poi_tops_shifted = poi_mle.cdf(count_tops[1:])
poi_bins = np.insert(poi_tops_shifted - poi_tops[:-1], 0, poi_tops[0])
#%% Prep for nn training                             


x_train, y_train, x_test, y_test = rlf.prep_count_data(counts_sim_df, bin_tops)

#%% Set up neural network
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
#%
hlayer_len = [100]
# Number of nodes in output layer: if series, 1, else number of cols
out_layer_len = 1 if len(y_train.shape)==1 else y_train.shape[1]
model = Sequential()
model.add(Dense(hlayer_len[0], input_shape=(x_train.shape[1],)))
model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                              # of the layer above. Here, with a "rectified linear unit",
                              # we clamp all values below 0 to 0.
                           
model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(hlayer_len[0]))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#model.add(Dense(hlayer_len[0]))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(hlayer_len[0]))
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Dense(out_layer_len))
model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probaility distribution, that is
                                 # that its values are all non-negative and sum to 1.

#%
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# For categorical target
model.compile(loss='categorical_crossentropy', optimizer=sgd)
#model.compile(loss='categorical_crossentropy', optimizer='adam')
# For numerical target
#model.compile(loss='mse', optimizer=sgd)
#model.get_config()
#%
model.fit(x_train, y_train,
          batch_size=32, nb_epoch=4,
          show_accuracy=True, verbose=1,
          validation_data=(x_test, y_test))

#model.get_weights()

#%%          
# Look at probabilities
proba = model.predict_proba(x_test, batch_size=32)
#%%
nn_probs = pd.DataFrame(proba, index = range(0,t_end-1), columns = )