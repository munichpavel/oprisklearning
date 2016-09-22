import risklearning.learning_severity as rl

days_year = 365

freq_param = 50
sev_param = {'logmu': 10.0, 'logsigma': 1.2}

n_years = 1
n_periods = days_year*n_years

l2_loss_events = rl.gen_loss_events(freq_param, sev_param, n_periods, 0,0)