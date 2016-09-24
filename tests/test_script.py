import numpy as np
import risklearning.learning_severity as rl
import NielsenNets.network as net
from statsmodels.distributions.empirical_distribution import ECDF


days_year = 365

#freq_param = 50
#sev_param = {'logmu': 10.0, 'logsigma': 1.2}

n_years = 3
n_periods = days_year*n_years

lambdas = [50.0, 200.0, 500.0, 5000.0] #Params for freq
ldays = [l/days_year for l in lambdas]
lmus = [15.0, 12.0, 10.0, 8.0]  # Params for sev
scales = np.exp(lmus) # For SciPy parametrization of lognormal
lsigs = [3.0, 2.0, 1.5, 1.2]

freq_params = ldays
sev_params = [{'logmu':lmus[p], 'logsigma':lsigs[p], 'scale': np.exp(lmus[p])} for p in range(0,4)]

l2_loss_events = rl.gen_loss_events(freq_params, sev_params, n_periods, 2,3)

# Select 2nd year to calibrate
test_year = 2
test_begin = (test_year - 1)*days_year + 1
test_end = (test_year)*days_year

train = l2_loss_events[l2_loss_events['when'] < test_begin]
train_p1 = l2_loss_events[(l2_loss_events['when'] >= test_begin) & (l2_loss_events['when'] <= test_end) ]

sev_ecdf = ECDF(train_p1['losses'])
train_y = sev_ecdf(train['losses'])

# Reshape input data to work with nn implementation
train_x1 = train.apply(lambda x: np.asarray(x), axis=1)
train_x = np.asarray(train_x1)

# Convert train_x and train_y into tuples as expected by SGD
train_xy = zip(train_x, train_y)


# Create network with 4 input nodes and 1 output nodes. Hidden nodes to be determined.
net = net.Network([4,4,1])
batch_size = 30
epochs = 10
eta = 3.0
net.SGD(train_xy, batch_size, epochs, eta)