rm(list = ls())
setwd("/home/pavel/Code/Python/risklearning/")
require(copula)
require(matrixcalc)
require(tidyr)
require(dplyr)

source('gen_losses_r/helpers.R')
###########################################################
# This script generates op risk loss data to compare
# trained neural networks to MLE. The loss data sets generated
# are non-stationary, and exhibit various dependency structures,
# such as Gaussian and Gumbel copulas.
# In future work, we may add a noise variable.
#
# For now, we only consider count data, i.e. how many events
# per day
############################################################

tenors_horizon = 365 # (Time) tenors (e.g. 1 day) per model horizon (e.g. 1 year)

h_start = 5.0 # How many model horizons of past data to train
h_end = 1.0 #How many model horizons of past data to test / validate

# Present is tenor 0, and boundary between training and testing data sets
t_start = -floor(h_start*tenors_horizon)
t_end = floor(h_end*tenors_horizon)

tenors = seq(t_start, t_end-1, by = 1)
n_tenors = t_end - t_start

####################################################
# Single count process
####################################################
l1_cats = 'EDPM'
l2_cats = 'TCEM'

or_cats = data.frame(L1_cat = l1_cats, L2_cat = l2_cats)
lambda_init = 1 # intensity over tenor (e.g. day)
lambda_final = 4 # intensity over tenor (e.g. day)

counts = gen_or_events(lambda_init, lambda_final, tenors, or_cats, '', 'tcem_1d')

#############################################################
# Gauss copula for 3 OpRisk types, EFTF, EFSS, TCEM,  where
# EFTF: External Fraud (Theft and Fraud)--think claims fraud
# ETSS: External Fraud (System Security)--think hack
# TCEM: Transaction Capture, Execution and Maintenance--think
#         fat-finger mistake
############################################################

# Define Gaussian copula with E
nc3 = normalCopula(param = c(0.5, 0.25, 0.25), dim=3, dispstr = 'un')


#% Generate Poisson-distributed events
l1_cats = c('EF', 'EF', 'EDPM')
l2_cats = c('EFTF', 'EFSS', 'TCEM')

or_cats = data.frame(L1_cat = l1_cats, L2_cat = l2_cats)
# External fraud remains stationary, while external fraud (system security) increases rapidly,
# and TCEM increases modestly
lambda_init = c(0.5, 0.1, 5) # intensity over tenor (e.g. day)
lambda_final = c(0.5, 1.5, 8) # intensity over tenor (e.g. day)

counts = gen_or_events(lambda_init, lambda_final, tenors, or_cats, nc3, 'gauss_3d')
