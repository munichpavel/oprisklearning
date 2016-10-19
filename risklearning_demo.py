
# coding: utf-8

# # risklearning demo
# 
# Most, if not all, operational risk capital models assume the existence of stationary frequency and severity distributions (typically Poisson for frequencies, and a subexponential distribution such as lognormal for severities). Yet every quarter (or whenever the model is recalibrated) risk capital goes up almost without fail, either because frequencies increase, severities increase or both.
# 
# The assumption of stationary distributions is just one limitation of current approaches to operational risk modeling, but it offers a good inroad for modeling approaches beyond the usual actuarial model typical in operational capital models.
# 
# In this notebook, we give a first example of how neural networks can overcome the stationarity assumptions of traditional approaches. The hope is that this is but one of many examples showing a better way to model operational risk.
# 
# Note: What follows if very much a work in progress . . .
# 
# 

# In[1]:

import risklearning.learning_frequency as rlf


# In[2]:

import pandas as pd
import numpy as np
import scipy.stats as stats
import math

import ggplot as gg

# ## Set up frequency distribution to generate samples

# In[3]:

tenors_horizon = 365 # (Time) tenors (e.g. 1 day) per model horizon (e.g. 1 year)

h_start = 5.0 # How many model horizons of past data to train
h_end = 1.0 #How many model horizons of past data to test / validate

# Present is tenor 0, and boundary between training and testing data sets
t_start = -int(math.floor(h_start*tenors_horizon))
t_end = int(math.floor(h_end*tenors_horizon))


#% Generate Poisson-distributed events
lambda_init = 1 # intensity over tenor (e.g. day)
lambda_final = 4 # intensity over tenor (e.g. day)
n_tenors = t_end - t_start
lambda_ts = np.linspace(lambda_init, lambda_final, num=n_tenors)
freq_rv = stats.poisson
counts = rlf.sim_counts(lambda_ts, freq_rv)

# Build df around counts, level 1 and 2 categorization of Operational Risk events
l1s = ['Execution Delivery and Process Management']*n_tenors
l2s = ['Transaction Capture, Execution and Maintenance']*n_tenors
tenors = list(xrange(t_start, t_end))

counts_sim_df = pd.DataFrame({'t': tenors,
                              'OR Category L1': l1s, 'OR Category L2': l2s,
                              'counts': counts})


# In[4]:


#%% Do MLE (simple average for Poisson process
n_tenors_train = -t_start
n_tenors_test = t_end

counts_train = (counts_sim_df[counts_sim_df.t < 0]).groupby('OR Category L2').sum()
counts_test =  (counts_sim_df[counts_sim_df.t >= 0]).groupby('OR Category L2').sum()


# ## MLE for training data
# 
# For the Poisson distribution, the MLE of the intensity (here lambda) is just the average of the counts per model horizon. In practice, OpRisk models sometimes take a weighted average, with the weight linearly decreasing over a period of years (see e.g. "LDA at Work" by Aue and Kalkbrener).

# In[5]:


lambdas_train = counts_train['counts']/n_tenors_train
lambdas_test = counts_train['counts']/n_tenors_test

bin_tops = [1,2,3,4,5,6,7,8,9,10,15,101]
# Recall that digitize (used later) defines bins by lower <= x < upper
count_tops =[count - 1 for count in bin_tops]

# Calculate bin probabilities from MLE poisson
poi_mle = stats.poisson(lambdas_train)
poi_bins = rlf.bin_probs(poi_mle, bin_tops)
#%%    
mle_probs = pd.DataFrame({'Count Top': count_tops, 'Probs': poi_bins})
#mle_probs = pd.DataFrame(poi_bins, index = [t-1 for t in bin_tops], columns = ['Prob'])
mle_probs.transpose()
mle_probs_vals = list(mle_probs.Probs)
# Visualize pdf (w.r.t. bins)
gg.ggplot(mle_probs, gg.aes(x='Count Top',weight='Probs')) \
    + gg.geom_bar()
# Note bug re: stat = 'identity' in ggplot: 
#   http://stackoverflow.com/questions/22599521/how-do-i-create-a-bar-chart-in-python-ggplot

#%% Compare to "true"
tenor = 0
true_poi_bins_0 = rlf.bin_probs(stats.poisson(lambda_ts[-t_start+tenor]), bin_tops)

true_probs_0 = pd.DataFrame({'Tenor': tenor, 'Count Top': count_tops, \
                            'Probs': true_poi_bins_0, 'Probs MLE': mle_probs_vals}, \
                            index = range(tenor*len(count_tops), \
                                    tenor*len(count_tops) + len(count_tops)))
                            #%%
tenor = t_end-1
true_poi_bins_1 = rlf.bin_probs(stats.poisson(lambda_ts[-t_start+tenor]), bin_tops)
true_probs_1 = pd.DataFrame({'Tenor': tenor, 'Count Top': count_tops, \
                            'Probs': true_poi_bins_1, 'Probs MLE': mle_probs_vals}, \
                            index = range(tenor*len(count_tops), \
                                    tenor*len(count_tops) + len(count_tops)))
                                    
#%% Now done below after NN
true_list = []
for t in range(0, t_end):
#for t in range(0, 2):
    true_poi_bins_t = rlf.bin_probs(stats.poisson(lambda_ts[-t_start+t]), bin_tops)
    true_probs_t = pd.DataFrame({'Tenor': t, 'Count Top': count_tops, \
                            'Probs': true_poi_bins_0, 'Probs MLE': mle_probs_vals}, \
                            index = range(t*len(count_tops), \
                                    t*len(count_tops) + len(count_tops)))
    true_list.append(true_probs_t)
    
true_probs = pd.concat(true_list)
                            

#%%
true_probs = pd.concat([true_probs_0, true_probs_1])
#%%
gg.ggplot(true_probs, gg.aes(x='Count Top',weight='Probs')) \
    + gg.facet_grid('Tenor') \
    + gg.geom_bar() \
    + gg.geom_step(gg.aes(y='Probs MLE')) \
    + gg.scale_x_continuous(limits = (0,len(count_tops)))
   # + gg.geom_bar(gg.aes(x='Count Top', weight='Probs MLE', position = 'dodge'))

#%%
# ## Prep simulated losses for neural network
# 
# For example
# 
# * Use one-hot-encoding for L1 and L2 categories (this will make more sense once we look at multiple dependent categories)
# * Bin count data
# * Normalize tenors (i.e. scale so that first tenor maps to -1 with 0 preserved)
# * Export as numpy arrays to feed into keras / tensorflow

# In[6]:

import warnings
warnings.filterwarnings('ignore') # TODO: improve slicing to avoid warnings

x_train, y_train, x_test, y_test = rlf.prep_count_data(counts_sim_df, bin_tops)


# ## Set up the network architecture and train
# 
# We use keras with TensorFlow backend.
# 
# Note: there has been no real attempt yet to optimize metaparameters.

# In[7]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

hlayer_len = [100] # As series in anticipation of different sized layers

# Number of nodes in output layer: if series, 1, else number of cols
out_layer_len = 1 if len(y_train.shape)==1 else y_train.shape[1]
model = Sequential()
model.add(Dense(hlayer_len[0], input_shape=(x_train.shape[1],)))
model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                              # of the layer above. Here, with a "rectified linear unit",
                              # we clamp all values below 0 to 0.
                           
model.add(Dropout(0.2))   # Default dropout parameter
model.add(Dense(hlayer_len[0]))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(hlayer_len[0]))
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Dense(out_layer_len))
model.add(Activation('softmax')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# For categorical target
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train,
          batch_size=32, nb_epoch=4,
          show_accuracy=True, verbose=1,
          validation_data=(x_test, y_test))


# ## Neural network frequency distribution
# 
# If the neural network has learned anything, we will see that the probility distribution shifts over time to higher buckets.

# In[8]:

proba = model.predict_proba(x_test, batch_size=32)
proba



# In[9]:

#%% Convert proba from wide to long and append to other probs
# TODO: Missing last tenor in nn proba (already in x_test, y_test)
probs_list = []

for t in range(proba.shape[0]):
    nn_probs_t = proba[t]    
    true_bins_t = rlf.bin_probs(stats.poisson(lambda_ts[-t_start+t]), bin_tops)
    probs_t = pd.DataFrame({'Tenor': t, 'Count Top': count_tops, \
                            'True Probs': true_bins_t, \
                            'Probs NN': nn_probs_t, \
                            'Probs MLE': mle_probs_vals}, \
                            index = range(t*len(count_tops), \
                                    t*len(count_tops) + len(count_tops)))
    probs_list.append(probs_t)

probs = pd.concat(probs_list)
#%%
probs_small = probs[probs.Tenor > 360 ]
#%%
gg.ggplot(probs_small, gg.aes(x='Count Top',weight='True Probs')) \
    + gg.facet_grid('Tenor') \
    + gg.geom_bar() \
    + gg.geom_step(gg.aes(y='Probs MLE', color = 'red')) \
    + gg.geom_step(gg.aes(y='Probs NN', color = 'blue')) \
    + gg.scale_x_continuous(limits = (0,len(count_tops)))

#    + gg.geom_step(gg.aes(y='Probs NN')) \ 


#%%

#nn_probs = pd.DataFrame(proba, index = range(0,t_end-1), columns = [t-1 for t in bin_tops])
# Heads (i.e. starting from present)
#nn_probs.head()


# In[10]:

# Tails (i.e. going to end of model horizon of 1 yr)
#nn_probs.tail()


# In[11]:

# And what MLE told us before
#mle_probs.transpose()


# ## Summary and next steps
# 
# We can see by the nn_probs data frame that the probability mass of the neural network shifts to the right, as does the underlying Poisson processes, with its intensity starting at 1 events per tenor / day at - 5 yrs and ending at 4 events per tenor / day at +1 yrs.
# 
# Next steps:
# 
# * Use better metric on generalization error that looking at probability tables (KS?)
# * Optimize hyperparameters
# * Simulate multiple, correlated Poisson processes
# * Test non-linear non-stationarities
# * Try recurrent neural network
# * Try convolution network
# 
# 
