#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ideas and code snippets borrowed from 

http://connor-johnson.com/2014/11/08/compound-poisson-processes/
and
https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
"""
from __future__ import division, print_function, absolute_import

import argparse
import sys
import logging

import scipy as sp
import numpy as np
import scipy.stats as stats
import pandas as pd
import math
#from risklearning import __version__

__author__ = "munichpavel"
__copyright__ = "munichpavel"
__license__ = "none"

_logger = logging.getLogger(__name__)
#%%

# Define distributions
poi = stats.poisson
lnorm = stats.lognorm 

# Parameters for simulation
n_years = 1
n_days = 365*n_years

lambdas = [50.0, 200.0, 500.0, 5000.0] #Params for freq
ldays = [l/n_days for l in lambdas]
lmus = [15.0, 12.0, 10.0, 8.0]  # Params for sev
sigs = [3.0, 2.0, 1.5, 1.2]



#%%
p_l1 = 0 # Dummy for now
p_l2 = 3 # Select which process to simulate
counts = poi(ldays[p_l2]).rvs(n_days)
s = 1
# Why are these so low???
losses_day = [lnorm(lsigs[p_l2],loc=lmus[p_l2]).rvs(N) for N in counts]
#%%
day_list = []

for d in range(0,n_days):
    days = [d]*losses_day[d].size
    process_l2 = [p_l2]*losses_day[d].size
    process_l1 = [p_l1]*losses_day[d].size
    day_list.append(pd.DataFrame({'losses': losses_day[d], 'when': days, 
                       'l1': process_l1, 'l2': process_l2 }))
                       
# Convert to dataframe
loss_events = pd.concat(day_list)

