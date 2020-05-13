
from sacred import Experiment
from datetime import datetime
ex = Experiment("Dynamic Fairness")

@ex.config
def my_config():
    """##Hyperparameter"""

    EXPERIMENT = 1 # Experiement i Corresponds to Figure i in the paper

    DATA_SET = 0
    PLOT_PREFIX = "plots/Exp{}/ ".format(EXPERIMENT)
    U_ALPHA = 0.5
    U_BETA = 0.5
    U_STD = 0.3
    w_fair = 10
    BI_LEFT = 0.5
    alpha = U_ALPHA
    beta = U_BETA
    u_std = U_STD
    KP = 0.01
    PROB_W = 5
    LP_COMPENSATE_W = 1# 0.025 #1
    HIDDEN_UNITS = 64
    eps_m = 0
    eps_p = 1
    trials = 2
    iterations = 3000

    MOVIE_RATING_FILE = "data/movie_data_binary_latent_5Comp_trial0.npy"
    movie_ranking_sample_file = "data/movie_data_binary_latent_5Comp_trial"
    n_user =10000
    n_movies = 100
    #n_companies = 5
    n_company = 5
    movie_features = "factorization"

    distribution = "bimodal" #User Distribution (bimodal, beta, discrete)
    if int(EXPERIMENT) < 7:
        DATA_SET = 0  #NewsData
    else:
        DATA_SET = 1  #Movie Data