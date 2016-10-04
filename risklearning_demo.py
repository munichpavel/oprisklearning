# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:16:49 2016

@author: pavel
"""

#%cd /home/pavel/Code/Python/risklearning

#%%
import pandas as pd
import numpy as np

from sklearn import preprocessing

import math

import risklearning.learning_frequency as rlf

#%%
# Read in loss data (counts per day / loss category)

days_year = 365
loss_ct_file = 'data/event_counts.csv'
loss_counts_raw = pd.read_csv(loss_ct_file)


#%%

## Restrict data
l1_sel = 'Clients Products and Business Practices'
l2_sel = ['Transaction Capture, Execution and Maintenance',
          'Suitability, Information Disclosure and Fiduciary Duty']

loss_counts_sel = loss_counts_raw[(loss_counts_raw['OR Category L2'] == l2_sel[0])
                                  | (loss_counts_raw['OR Category L2'] == l2_sel[1])]
# loss_counts_sel = loss_counts_raw

t_start = -math.floor(2*days_year)
t_end = math.floor(days_year/2)

# TODO check col names, or add col name as argument
loss_counts_sel = loss_counts_sel[(loss_counts_sel['current_delta'] >= t_start)
                                &(loss_counts_sel['current_delta'] < t_end)]
#loss_counts_sel = loss_counts_sel[loss_counts_sel['current_delta'] < t_end]

#
bin_tops = [1,2, 15]

x_train, y_train, x_test, y_test = rlf.prep_count_data(loss_counts_sel, bin_tops)
#%% Set up neural network
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
#%%
hlayer_len = [100]
model = Sequential()
model.add(Dense(hlayer_len[0], input_shape=(x_train.shape[1],)))
model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                              # of the layer above. Here, with a "rectified linear unit",
                              # we clamp all values below 0 to 0.
                           
model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(hlayer_len[0]))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(y_test.shape[1]))
model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probaility distribution, that is
                                 # that its values are all non-negative and sum to 1.

#%%
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
#%%
model.fit(x_train, y_train,
          batch_size=32, nb_epoch=50,
          show_accuracy=True, verbose=1,
          validation_data=(x_test, y_test))

# Look at probabilities
proba = model.predict_proba(x_test, batch_size=32)