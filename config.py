
from sacred import Experiment
from datetime import datetime
ex = Experiment("Dynamic Fairness")

@ex.config
def my_config():
    """##Hyperparameter"""
    DATA_SET = 0 #0 Synthetic old, 1 Jokes, 2 Movies
    PLOT_PREFIX = "plots/"
    U_ALPHA = 0.5
    U_BETA = 0.5
    U_STD = 0.3
    w_fair = 10
    BI_LEFT = 0.5
    alpha = U_ALPHA
    beta = U_BETA
    std= U_STD
    KP = 0.01
    PROB_W = 5
    LP_COMPENSATE_W = 1# 0.025 #1
    HIDDEN_UNITS = 64
    eps_m = 0
    eps_p = 1

    MOVIE_RATING_FILE = "data/movie_data_binary_latent_5Comp_trial0.npy"
    movie_ranking_sample_file = "data/movie_data_binary_latent_5Comp_trial"
    n_users =10000
    n_movies = 100
    n_companies = 5
    movie_features = "factorization"

    distribution = "bimodal" #User Distribution (bimodal, beta, discrete)