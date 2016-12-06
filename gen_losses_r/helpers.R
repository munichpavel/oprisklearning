# Helper files to generate OR loss data
sample_pois_counts = function(lambda_vec, copula){
  n_lambda = length(lambda_vec)
  param_list = lapply(1:n_lambda, function(j) list(lambda = lambda_vec[j]))
  norm_poisson = mvdc(copula, rep('pois', n_lambda), param_list)
  return(rMvdc(1, norm_poisson))
}

# Add L1 category via L2 category
assign_l1 = function(l2, or_cats){
  return(or_cats$L1_cat[which(or_cats$L2_cat == l2)])
}

# 
gen_or_events = function(lambda_init, lambda_final, tenors, or_cats, copula, out_stem){
  
  n_tenors = length(tenors)
  n_cats = dim(or_cats)[1]
  # Make linear grid from lambda_init to lambda_final
  lambdas = sapply(1:length(lambda_init), function(i) seq(lambda_init[i], lambda_final[i], length.out = n_tenors))
  
  if (typeof(copula) == 'S4'){ # Check if S4 class, hopefully copula
    counts_df = data.frame(t(apply(lambdas, MARGIN = 1,
                                   FUN = function(lambda_vec) sample_pois_counts(lambda_vec, nc3))))
  } else { # univariate distribution
    counts_df = data.frame(sapply(lambdas, function(lambda) rpois(1, lambda)))
  }
  colnames(counts_df) = l2_cats
  # Add column for tenor
  counts_df$t = tenors
  
  # Gather
  counts_long = counts_df %>% gather(L2_cat, counts, 1:n_cats) %>% arrange(t)
  
  
  l1s = sapply(counts_long$L2, function(l2){assign_l1(l2, or_cats)})
  counts_long$L1_cat = l1s
  counts_out = counts_long[, c('t', 'L1_cat', 'L2_cat', 'counts')]
  # Write to tab-delimited file
  write.csv(counts_out, file = paste0('data/', out_stem, '.csv'), row.names = F)
  return(counts_out)
  
}