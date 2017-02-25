
# coding: utf-8

# # risklearning demo 2
# 
# In the second risklearning demo, we consider multidimensional non-stationary Poisson processes with dependence structure given by a Gaussian copula.

# In[3]:

import risklearning.learning_frequency as rlf

reload(rlf)


# In[4]:

import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import matplotlib.style
matplotlib.style.use('ggplot')
import ggplot as gg
#get_ipython().magic(u'matplotlib inline')


# ## Set up frequency distribution to generate samples

# In[35]:

# Read in Poisson parameters used to simulate loss counts
lambdas_df = pd.read_csv('data/lambdas_gauss_3d.csv')
lambda_start = lambdas_df.head(1).iloc[0]
lambda_end = lambdas_df.tail(1).iloc[0]
print 'Initial lambdas:\n{}'.format(lambda_start)
print '\nFinal lambdas: \n{}'.format(lambda_end)

#%%
lambda_start = lambdas_df['TCEM'][0]
lambda_end = lambdas_df['TCEM'].tail(1).iloc[0]
print('Lambda start value: {}, lambda end value: {}'.format(lambda_start, lambda_end))
lambda_ts = lambdas_df['TCEM']
# Read in simulated loss counts
counts_sim_df = pd.read_csv('data/tcem_1d.csv')
# EDPM: Execution, Delivery and Process Management
# TCEM: Transaction Capture, Execution and Maintenance--think fat-finger mistake
counts_sim_df.head()


# In[52]:

#%% Do MLE (simple average for Poisson process
t_start = np.min(counts_sim_df['t'])
t_end = np.max(counts_sim_df['t'])

n_tenors_train = -t_start
n_tenors_test = t_end

counts_train = (counts_sim_df[counts_sim_df.t < 0]).groupby('L2_cat').sum()
counts_test =  (counts_sim_df[counts_sim_df.t >= 0]).groupby('L2_cat').sum()


# ## MLE for training data
# 
# For the Poisson distribution, the MLE of the intensity (here lambda) is just the average of the counts per model horizon. In practice, OpRisk models sometimes take a weighted average, with the weight linearly decreasing over a period of years (see e.g. "LDA at Work" by Aue and Kalkbrener).

# In[31]:

lambdas_train = counts_train['counts']/n_tenors_train
lambdas_test = counts_test['counts']/n_tenors_test

bin_tops = [1,2,3,4,5,6,7,8,9,10,15,101]
# Recall that digitize (used later) defines bins by lower <= x < upper
count_tops =[count - 1 for count in bin_tops]

# Calculate bin probabilities from MLE poisson
poi_mle = stats.poisson(lambdas_train)
poi_bins = rlf.bin_probs(poi_mle, bin_tops)

mle_probs = pd.DataFrame({'Count Top': count_tops, 'Probs': poi_bins})
# For later comparison
mle_probs_vals = list(mle_probs.Probs)


# ## Prep simulated losses for neural network
# 
# For example
# 
# * Use one-hot-encoding for L1 and L2 categories (this will make more sense once we look at multiple dependent categories)
# * Bin count data
# * Normalize tenors (i.e. scale so that first tenor maps to -1 with 0 preserved)
# * Export as numpy arrays to feed into keras / tensorflow

# In[32]:

import warnings
warnings.filterwarnings('ignore') # TODO: improve slicing to avoid warnings

x_train, y_train, x_test, y_test = rlf.prep_count_data(counts_sim_df, bin_tops)

## With tensorflow
import tensorflow as tf

layer_width = [y_train.shape[1]]
x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
W = tf.Variable(tf.zeros([x_train.shape[1], layer_width[0]]))
b = tf.Variable(tf.zeros([layer_width[0]]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, y_train.shape[1]])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess


# ## Set up the network architecture and train
# 
# We use keras with TensorFlow backend. Later we will look at optimizing metaparameters.
# 

# In[33]:

#from keras.optimizers import SGD
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# rl_train_net is a wrapper for standard keras functionality that
# makes it easier below to optimize hyperparameters
rl_net = rlf.rl_train_net(x_train, y_train, x_test, y_test, [150],                     n_epoch = 300, optimizer = 'adagrad')
proba = rl_net['probs_nn']


# ## Evaluating the neural network
# Let's see now how the neural network tracks the true distribution over time, and compare with the MLE fitted distribution.
# 
# We do this both numerically (Kullback-Leibler divergance) and graphically.

# In[43]:

#% Convert proba from wide to long and append to other probs
mle_probs_vals = list(mle_probs.Probs)
# TODO: Missing last tenor in nn proba (already in x_test, y_test)
probs_list = []
kl_mle_list = []
kl_nn_list = []

for t in range(proba.shape[0]):
    nn_probs_t = proba[t]    
    true_bins_t = rlf.bin_probs(stats.poisson(lambda_ts[-t_start+t]), bin_tops)
    probs_t = pd.DataFrame({'Tenor': t, 'Count Top': count_tops,                             'Probs True': true_bins_t,                             'Probs NN': nn_probs_t,                             'Probs MLE': mle_probs_vals},                             index = range(t*len(count_tops),                                     t*len(count_tops) + len(count_tops)))
    probs_list.append(probs_t)
    # Calculate KL divergences
    kl_mle_list.append(stats.entropy(true_bins_t, mle_probs_vals))
    kl_nn_list.append(stats.entropy(true_bins_t, nn_probs_t))

probs = pd.concat(probs_list)


# In[44]:

probs_tail = probs[probs.Tenor > 360 ]

gg.ggplot(probs_tail, gg.aes(x='Count Top',weight='Probs True'))     + gg.facet_grid('Tenor')     + gg.geom_bar()     + gg.geom_step(gg.aes(y='Probs MLE', color = 'red'))     + gg.geom_step(gg.aes(y='Probs NN', color = 'blue'))     + gg.scale_x_continuous(limits = (0,len(count_tops)))


# In[57]:

# KL divergences

kl_df = pd.DataFrame({'Tenor': range(0, t_end+1),                       'KL MLE': kl_mle_list,                       'KL NN': kl_nn_list})

print kl_df.head()
print kl_df.tail()                      
#%                      
# Plot KL divergences
gg.ggplot(kl_df, gg.aes(x='Tenor'))     + gg.geom_step(gg.aes(y='KL MLE', color = 'red'))     + gg.geom_step(gg.aes(y='KL NN', color = 'blue'))


# # Optimizing network architecture

# In[61]:

# More systematically with NN architecture
# Loop over different architectures, create panel plot
neurons_list = [10, 20,50,100, 150, 200]
#neurons_list = [10, 20,50]
depths_list = [1,2,3]
optimizer = 'adagrad'
#%%
kl_df_list = []
for depth in depths_list:
    for n_neurons in neurons_list:
        nn_arch = [n_neurons]*depth
        print("Training " + str(depth) + " layer(s) of " + str(n_neurons) + " neurons")
        rl_net = rlf.rl_train_net(x_train, y_train, x_test, y_test, nn_arch,                     n_epoch = 300, optimizer = optimizer)
        proba = rl_net['probs_nn']
        print("\nPredicting with " + str(depth) + " layer(s) of " + str(n_neurons) + " neurons")
        probs_kl_dict = rlf.probs_kl(proba, lambda_ts, t_start, t_end+1, bin_tops, mle_probs_vals)
        probs = probs_kl_dict['Probs']
        kl_df_n = probs_kl_dict['KL df']
    
        kl_df_n['Hidden layers'] = depth
        kl_df_n['Neurons per layer'] = n_neurons
        kl_df_n['Architecture'] = str(depth) + '_layers_of_' + str(n_neurons)             + '_neurons'

        kl_df_list.append(kl_df_n)
 #%%
kl_df_hyper = pd.concat(kl_df_list)


# In[62]:

# Plot
kl_mle = kl_df_n['KL MLE'] # These values are constant over the above loops (KL between MLE and true distribution)
for depth in depths_list:
    kl_df_depth = kl_df_hyper[kl_df_hyper['Hidden layers'] == depth]
    kl_df_depth = kl_df_hyper[kl_df_hyper['Hidden layers'] == depth]
    kl_depth_vals = kl_df_depth.pivot(index = 'Tenor', columns = 'Neurons per layer', values = 'KL NN')
    kl_depth_vals['KL MLE'] = kl_mle
    kl_depth_vals.plot(title = 'Kullback-Leibler divergences from true distribution \n for '                        + str(depth) + ' hidden layer(s)',                       figsize = (16,10))


# In[65]:

# Try again, but now with RMSprop
neurons_list = [10, 20,50]
#neurons_list = [50]
depths_list = [2,3]
optimizer = 'RMSprop'
#%%
kl_df_list = []
for depth in depths_list:
    for n_neurons in neurons_list:
        nn_arch = [n_neurons]*depth
        print("Training " + str(depth) + " layer(s) of " + str(n_neurons) + " neurons")
        rl_net = rlf.rl_train_net(x_train, y_train, x_test, y_test, nn_arch,                     n_epoch = 300, optimizer = optimizer)
        proba = rl_net['probs_nn']
        print("\nPredicting with " + str(depth) + " layer(s) of " + str(n_neurons) + " neurons")
        probs_kl_dict = rlf.probs_kl(proba, lambda_ts, t_start, t_end+1, bin_tops, mle_probs_vals)
        probs = probs_kl_dict['Probs']
        kl_df_n = probs_kl_dict['KL df']
    
        kl_df_n['Hidden layers'] = depth
        kl_df_n['Neurons per layer'] = n_neurons
        kl_df_n['Architecture'] = str(depth) + '_layers_of_' + str(n_neurons)             + '_neurons'

        kl_df_list.append(kl_df_n)
 #%%
kl_df_hyper = pd.concat(kl_df_list)

# Plot
kl_mle = kl_df_n['KL MLE'] # These values are constant over the above loops (KL between MLE and true distribution)
for depth in depths_list:
    kl_df_depth = kl_df_hyper[kl_df_hyper['Hidden layers'] == depth]
    kl_df_depth = kl_df_hyper[kl_df_hyper['Hidden layers'] == depth]
    kl_depth_vals = kl_df_depth.pivot(index = 'Tenor', columns = 'Neurons per layer', values = 'KL NN')
    kl_depth_vals['KL MLE'] = kl_mle
    kl_depth_vals.plot(title = 'Kullback-Leibler divergences from true distribution \n for '                        + str(depth) + ' hidden layer(s)',                       figsize = (16,10))


# Note that with 50 nodes per layer, the KL error for RBM Neural Networks is worse than MLE once we are more than 100 tenors (here, days) from the beginning of the test sample. With more nodes per layer, the results are even worse, though we do not show them here.

# ## Summary and next steps
# 
# We can see by the nn_probs data frame that the probability mass of the neural network shifts to the right, as does the underlying Poisson processes, with its intensity starting at 1 events per tenor / day at - 5 yrs and ending at 4 events per tenor / day at +1 yrs.
# 
# Next steps:
# 
# * Simulate multiple, correlated Poisson processes
# * Test different optimizers
# * Test non-linear non-stationarities
# * Try recurrent neural network (?)
# * Try convolution network (?)
# 
# 
