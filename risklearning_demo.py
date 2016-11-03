
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

# In[25]:

import risklearning.learning_frequency as rlf
reload(rlf)


# In[26]:

import pandas as pd
import numpy as np
import scipy.stats as stats
import math

import ggplot as gg
get_ipython().magic(u'matplotlib inline')


# ## Set up frequency distribution to generate samples

# In[27]:

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


# In[28]:


#%% Do MLE (simple average for Poisson process
n_tenors_train = -t_start
n_tenors_test = t_end

counts_train = (counts_sim_df[counts_sim_df.t < 0]).groupby('OR Category L2').sum()
counts_test =  (counts_sim_df[counts_sim_df.t >= 0]).groupby('OR Category L2').sum()


# ## MLE for training data
# 
# For the Poisson distribution, the MLE of the intensity (here lambda) is just the average of the counts per model horizon. In practice, OpRisk models sometimes take a weighted average, with the weight linearly decreasing over a period of years (see e.g. "LDA at Work" by Aue and Kalkbrener).

# In[29]:

lambdas_train = counts_train['counts']/n_tenors_train
lambdas_test = counts_train['counts']/n_tenors_test

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

# In[30]:

import warnings
warnings.filterwarnings('ignore') # TODO: improve slicing to avoid warnings

x_train, y_train, x_test, y_test = rlf.prep_count_data(counts_sim_df, bin_tops)


# ## Set up the network architecture and train
# 
# We use keras with TensorFlow backend. Later we will look at optimizing metaparameters.
# 

# In[32]:

#from keras.optimizers import SGD
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# rl_train_net is a wrapper for standard keras functionality that
# makes it easier below to optimize hyperparameters
rl_net = rlf.rl_train_net(x_train, y_train, x_test, y_test, [150],                     n_epoch = 200, optimizer = 'adagrad')


# ## Neural network frequency distribution
# 
# If the neural network has learned anything, we will see that the probility distribution shifts over time to higher buckets.

# In[33]:

proba = rl_net['probs_nn']


# In[34]:

nn_probs = pd.DataFrame(proba, index = range(0,t_end), columns = [t-1 for t in bin_tops])
# Heads (i.e. starting from present)
nn_probs.head()


# In[35]:

# Tails (i.e. going to end of model horizon of 1 yr)
nn_probs.tail()


# In[36]:

# And what MLE told us before
mle_probs.transpose()


# ## Evaluating the neural network
# The above shows that the neural network learns that counts increase over time, but we want more than just the correct trend, we want to see how far the neural network is from the true distribution, and compare with the MLE fitted distribution.
# 
# We do this both numerically (Kullback-Leibler divergance) and graphically.

# In[37]:

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


# In[38]:

probs_tail = probs[probs.Tenor > 360 ]

gg.ggplot(probs_tail, gg.aes(x='Count Top',weight='Probs True'))     + gg.facet_grid('Tenor')     + gg.geom_bar()     + gg.geom_step(gg.aes(y='Probs MLE', color = 'red'))     + gg.geom_step(gg.aes(y='Probs NN', color = 'blue'))     + gg.scale_x_continuous(limits = (0,len(count_tops)))



# In[39]:

# KL divergences

kl_df = pd.DataFrame({'Tenor': range(0, t_end),                       'KL MLE': kl_mle_list,                       'KL NN': kl_nn_list})

print kl_df.head()

print kl_df.tail()                      
#%                      
# Plot KL divergences
gg.ggplot(kl_df, gg.aes(x='Tenor'))     + gg.geom_step(gg.aes(y='KL MLE', color = 'red'))     + gg.geom_step(gg.aes(y='KL NN', color = 'blue'))


# # Optimizing network architecture

# In[48]:

# More systematically with NN architecture
# Loop over different architectures, create panel plot
#neurons_list = [10, 20,50,100, 200]
neurons_list = [10, 20,50]
depths_list = [1,2,3]
optimizer = 'adagrad'
#%%
kl_df_list = []
for depth in depths_list:
    for n_neurons in neurons_list:
        nn_arch = [n_neurons]*depth
        print("Training " + str(depth) + " layer(s) of " + str(n_neurons) + " neurons")
        rl_net = rlf.rl_train_net(x_train, y_train, x_test, y_test, nn_arch,                     n_epoch = 2, optimizer = optimizer)
        proba = rl_net['probs_nn']
        print("\nPredicting with " + str(depth) + " layer(s) of " + str(n_neurons) + " neurons")
        probs_kl_dict = rlf.probs_kl(proba, lambda_ts, t_start, t_end, bin_tops, mle_probs_vals)
        probs = probs_kl_dict['Probs']
        kl_df_n = probs_kl_dict['KL df']
    
        kl_df_n['Hidden layers'] = depth
        kl_df_n['Neurons per layer'] = n_neurons
        kl_df_n['Architecture'] = str(depth) + '_layers_of_' + str(n_neurons)             + '_neurons'

        kl_df_list.append(kl_df_n)
 #%%
kl_df_hyper = pd.concat(kl_df_list)


# In[52]:

# Plot
plot_file_stem = '/home/pavel/Code/Python/risklearning/plots/'
for depth in depths_list:
    kl_df_depth = kl_df_hyper[kl_df_hyper['Hidden layers'] == depth]
    kl_plot = gg.ggplot(kl_df_depth, gg.aes(x='Tenor'))         + gg.geom_point(gg.aes(y='KL MLE', color = 'red'))         + gg.geom_point(gg.aes(y='KL NN', color = 'Neurons per layer'))         + gg.ggtitle('Architecture: ' + str(depth) + ' hidden layer(s)')
    print(kl_plot)
    kl_plot_name = plot_file_stem + 'kl_plot_' + str(depth) + 'deep_opt_' + optimizer + '.png'
    gg.ggsave(kl_plot, kl_plot_name)


# ## Summary and next steps
# 
# We can see by the nn_probs data frame that the probability mass of the neural network shifts to the right, as does the underlying Poisson processes, with its intensity starting at 1 events per tenor / day at - 5 yrs and ending at 4 events per tenor / day at +1 yrs.
# 
# Next steps:
# 
# * Simulate multiple, correlated Poisson processes
# * Test non-linear non-stationarities
# * Try recurrent neural network
# * Try convolution network
# 
# 
