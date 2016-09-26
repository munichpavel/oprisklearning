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

from statsmodels.distributions.empirical_distribution import ECDF
#from risklearning import __version__

__author__ = "munichpavel"
__copyright__ = "munichpavel"
__license__ = "none"

_logger = logging.getLogger(__name__)
#%%


def sim_losses(freq_param, sev_param_initial, sev_param_final, n_tenors):
    """ 
    Simulate losses from actuarial model.
    At present, the loss process is a compound Poisson process with 
    lognormal severity distribution
    
    """
    #


    freq_rv = stats.poisson(freq_param) 
    #sev_rv = stats.lognorm(sev_param['logmu'], scale = sev_param['logsigma'])
    #sev_rv = stats.lognorm(sev_param['logsigma'], scale=sev_param['scale'])
    counts = freq_rv.rvs(n_tenors)

    # Generate severity parameters as function of time
    scale_t = np.linspace(sev_param_initial['scale'], sev_param_final['scale'], num=n_tenors)
    lsig_t = np.linspace(sev_param_initial['logsigma'], sev_param_final['logsigma'], num=n_tenors)

    losses_period = [stats.lognorm(lsig_t[t], scale=scale_t[t]).rvs(counts[t]) for t in xrange(n_tenors)]

#    losses_period = [stats.lognorm(lsig_t[t], scale = scale_t[t]).rvs(t) for t in xrange(n_tenors)]
    return(losses_period)


def gen_loss_events(freq_params, sev_params_initial, sev_params_final, n_tenors, process_l1, process_l2):
    """
    Generate loss events and return as data frame
    """
    #
    freq_param = freq_params[process_l2]
    sev_param_initial = sev_params_initial[process_l2]
    sev_param_final = sev_params_final[process_l2]

    loss_amts = sim_losses(freq_param, sev_param_initial, sev_param_final, n_tenors)

    # For each period (typically a day), augment loss amounts with
    # time of occurrence and process categories (levels 1 and 2)
    period_list = []
    for p in xrange(n_tenors):
        N = loss_amts[p].size  # Number of losses in that period / day
        ps = [p] * N
        l1s = [process_l1] * N
        l2s = [process_l2] * N
        period_list.append(pd.DataFrame({'losses': loss_amts[p], 'when': ps,
                                         'l1': l1s, 'l2': l2s}))

    loss_df = pd.concat(period_list, ignore_index=True)
    #
    return (loss_df)
