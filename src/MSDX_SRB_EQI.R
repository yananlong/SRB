# Load libraries
library(tidyverse)
library(brms)

# Set number of cores
options(mc.cores = parallel::detectCores())

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
num <- as.numeric(args[1]) # 1 - 125
reps <- as.numeric(args[2]) # default 10

# Base directory
namebase <- "data/"

# File I/O
## Load data and change column data types
df_fips_neighbor <- 
    read_delim(
        file = str_c(namebase, 'fips_with_neighbors.txt'),
        delim = "\n", col_names = "fips", col_types = "c"
    )f

df_srb_eqi <-
    read_delim(
        file = str_c(namebase, 'MSDX_day_20180922_SRB_clus_sept.csv'),
        delim = ","
    ) %>%
    select(-"county_name") %>%
    mutate_at(
        c("stfips", "state"),
        as.factor
    ) %>%
    mutate_at(c("F", "M"), as.integer) %>%
    mutate_if(is.double, as.factor) %>%
    filter(stfips %in% df_fips_neighbor$fips)

## Factor names
facs <-
    df_srb_eqi %>%
    select(-c(stfips:state)) %>%
    colnames()
fac <- facs[num]
cat(sprintf("Factor %d: %s\n", num, fac))

## Output files
### Model summary
fn_sum_state <- sprintf("%s%s/%s_%s_state.out", namebase, "KFOLDCV", "Summ", fac)
fn_kfold <- sprintf("%s%s/%s_%s.out", namebase, "KFOLDCV", "KFOLD_random", fac)

# Fit models
## Null
mod_null <- brm(formula = M | trials(M + F) ~ 1,
                data = df_srb_eqi,
                family = binomial("logit"),
                cores = getOption("mc.cores"),
                chains = getOption("mc.cores"),
                warmup = 500, iter = 1500,
                thin = 2, inits = 0)
print(mod_null, digits = 6)

## One factor
params <- 'data = df_srb_eqi, family = binomial("logit"),
           cores = getOption("mc.cores"), chains = getOption("mc.cores"),
           warmup = 500, iter = 1500, thin = 2, inits = 0'

## One factor plus state-level effects
mod_state_name <- sprintf("mod_%s_state", fac)
formula_state <- sprintf("M | trials(M + F) ~ 1 + %s + (1 | state)", fac)
brm_expr_state <- sprintf("%s <- brm(%s, %s)", mod_state_name, formula_state, params)
eval(parse(text = brm_expr_state))

### Write model summaries to output
sink(fn_sum_state, append = FALSE)
print(get(mod_state_name), digits = 6)
sink()

# Cross validation
## Compute IC and compare models
for (k in c(1:reps)){
    kfold_null <- kfold(mod_null)
    kfold_fac_state <- eval(parse(text = sprintf('kfold(%s)', mod_state_name)))
    kfoldcv <- loo_compare(kfold_null, kfold_fac_state)
    
    # Write to output
    kfoldcv %>%
        as_tibble(pkgconfig::set_config("tibble::rownames" = NA)) %>%
        rownames_to_column(var = "model") %>%
        select(c("model":"se_diff")) %>%
        write_delim(
            fn_kfold, delim = "\t",
            append = ifelse(k == 1, FALSE, TRUE),
            col_names = ifelse(k == 1, TRUE, FALSE)
        )
}
