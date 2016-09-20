# Set distribution data
require(dplyr)
#lambdas <- c(2,5,20, 500)
lambdas <- c(50, 200, 500, 5000)
lmus <- c(15, 12, 10, 8)
lsigmas <- c(3,2,1.5, 1.2)

                        
cat1s <- c(1,2,3,3)
cat2s <- 1:4

l_day <- lambdas/365

params_df <- data.frame(lambdas = l_day, lmus = lmus, lsigs = lsigmas, cat1s = cat1s, cat2s = cat2s)
                        
# Copula: TODO when I have internet access again
# TODO: Use time-series objects instead?

n_years <- 3
n_days <- 365*n_years

# Simulate daily losses
# Loss event data frame
le_df_empty = data.frame(loss = numeric(0), when = numeric(0), cat1 = numeric(0), cat2 = numeric(0))

# Store losses in list over n_days, with each day getting its own le_df
le_list = lapply(1:n_days, function(i) list())

i <- 4

# TODO with copula

d <- 365
le_list[[d]][[i]]<- le_df_d

# gen_le <- function(i,Ns_i, d, params_df){
#   if (Ns[d] > 0){
#     X_d <- rlnorm(Ns[d], params_df$lmus[i], params_df$lsigs[i])
#     le_df_d <- data.frame(loss = X_d, when = d, cat1 = params_df$cat1s[i], cat2 = params_df$cat2s[i])
#     return(le_df_d)
#   } else{
#     return(le_df_empty) 
#   }
# }

# Generate all Ns upfront
Ns <- sapply(1:4, function(i){
  Ns_i <- rpois(n_days, params_df$lambdas[i])
  Ns_i
})

gen_le <- function(i,N_id, d, params_df){
  if (N_id > 0){
    X_d <- rlnorm(N_id, params_df$lmus[i], params_df$lsigs[i])
    le_df_d <- data.frame(loss = X_d, when = d, cat1 = params_df$cat1s[i], cat2 = params_df$cat2s[i])
    return(le_df_d)
  } else{
    return(le_df_empty) 
  }
}


# Want a dataframe per day
les_all <- lapply(1:n_days, function(d){
  les_d <- lapply(1:4,function(i) gen_le(i, Ns[d,i], d, params_df) )
  # Return loss events on day d for all processes as dataframe
  rbind(les_d[[1]],
        les_d[[2]],
        les_d[[3]],
        les_d[[4]])
})


# Has to be better way
les_all_df <- le_df_empty
for (d in 1:n_days){
 les_all_df <- rbind(les_all_df, les_all[[d]]) 
}

# For fixed d define les up to and includeing day d, and losses for the year following day d
d <- 365
# Make when relative to d
les_all_d <- les_all_df %>% mutate(when_rel = when - d)
les_all_d$when <- NULL
les_pre_d <- les_all_d %>% filter(when_rel <= 0)
les_d_plus_1yr <- les_all_d %>% filter(when_rel > 0, when_rel <= 365)
