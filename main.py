"""
Controlling Fairness and Bias in Dynamic Learning-to-Rank
#TODO Readme
author: Marco Morik
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config import ex
import numpy as np
from scipy.stats import truncnorm
import scipy.integrate
import scipy.stats
import random
#from sinkhorn_knopp import sinkhorn_knopp as skp
import pandas as pd
import time
import warnings; warnings.simplefilter('ignore') ##Ignores Warnings for nicer Plots. Disable for Debugging
import data_utils
import os
import birkhoff
import relevance_network
from itertools import permutations

from Documents import Item, Movie, Joke
import plotting
plotting.init_plotting()

birkhoff.TOLERANCE = 10**(-8)


def __main__():
    global PLOT_PREFIX
    global sample_user
    global DATA_SET
    # overview = test_ranking(items, popularity, trials, click_models=["PBM_log"], methods=["Naive", "IPS", "P-Controller"], save = True, iterations=5000)
    # print(tabulate(overview, headers='keys', tablefmt='psql'))

    # experiment_different_starts(True)
    # load_and_plot_all()
    # test_different_population(10, 20000, True)
    # test_different_groups(40, 2000,False) #Maybe redo with more trials.
    # load_and_plot_lambda_comparison() #Runs the plotting of the Controller/LP for different lambdas

    # news_experiment()
    # load_and_plot_news()
    """
    movie_experiment("Movie_data/plots/Incomplete_LatentF/", ["Naive", "IPS", "Pers", "Skyline-Pers", "Fair-I-IPS", "Fair-E-IPS"], "data/movie_data_prepared_latent_features.npy", trials=5,iterations=8000)
    movie_experiment("Movie_data/plots/Full_LatentF/", ["Naive", "IPS", "Pers", "Skyline-Pers", "Fair-I-IPS", "Fair-E-IPS"],"data/movie_data_full_latent_features.npy", trials=5, iterations=8000)
    movie_experiment("Movie_data/plots/Full_120F/",
                     ["Naive", "IPS", "Pers", "Skyline-Pers", "Fair-I-IPS", "Fair-E-IPS"],
                     "data/movie_data_full_120features.npy", trials=5, iterations=8000)
    movie_experiment("Movie_data/plots/Incomplete_20F/", ["Naive", "IPS", "Pers", "Skyline-Pers", "Fair-I-IPS", "Fair-E-IPS"],"data/movie_data_prepared_20features.npy", trials=5, iterations=8000)
    movie_experiment("Movie_data/plots/Incomplete_120F/",
                     ["Naive", "IPS", "Pers", "Skyline-Pers", "Fair-I-IPS", "Fair-E-IPS"],
                     "data/movie_data_prepared_120features.npy", trials=5, iterations=8000)

    movie_experiment("Movie_data/plots/Latent_Full_NoNoise_NoClickModel/",
                     ["Naive", "IPS", "Pers", "Skyline-Pers", "Fair-I-IPS", "Fair-E-IPS"],
                     "data/movie_data_full_latent_features2.npy", trials=1, iterations=2000)

    movie_experiment("Movie_data/plots/BinaryMovies/",
                     ["Naive", "IPS", "Fair-I-IPS"],
                     # ["IPS", "Pers"],
                     "data/movie_data_binary_latent_trial0.npy",
                     # "data/movie_data_prepared_20features.npy",
                     trials=10, iterations=6000, binary_rel=True)


    """
    movie_experiment("Movie_data/plots/MovieExperimentFull_5Comanies_onlyHighestRatings/",
                     # "Movie_data/plots/MovieExperimentFull_binary/",
                     ["Naive", "IPS", "Pers", "Skyline-Pers", "Fair-I-IPS"],
                     # ["IPS", "Pers"],
                     # "data/movie_data_binary_latent_5Comp_trial0.npy",
                     "data/movie_data_binary_latent_5Comp_MostRatings_trial0.npy",
                     # "data/movie_data_prepared_20features.npy",
                     trials=1, iterations=6000, binary_rel=True)
    """
    movie_experiment("Movie_data/plots/BinaryMovies_Debug_5Comp/",
                     ["Pers",  "IPS"], # "Skyline-Pers"
                     # ["IPS", "Pers"],
                     "data/movie_data_binary_latent_5Comp_trial0.npy",
                     # "data/movie_data_prepared_20features.npy",
                     trials=1, iterations=6000, binary_rel=True)
    """
    # load_and_plot_movie_experiment()
    # DATA_SET = 0
    # test_different_groups(20, 3000, True, prefix="News_data/different_groups_binary_final2/")
    # experiment_different_starts(True, "News_data/different_starts_binary_final2/", news_items=True)
    # test_different_population(20, 3000, True, prefix="News_data/different_population_binary_final2/", news_items=True)

    """
    return
    PLOT_PREFIX = "News_data/plots/LambdaFinal_4/"

    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    trials = 20
    iterations = 3000
    multiple_items = [load_news_items(n=30, completly_random=True) for i in range(trials)]
    items = multiple_items[0]
    popularity = np.ones(len(items))
    G = assign_groups(items)
    click_models = ["lambda0","lambda0.0001", "lambda0.001", "lambda0.01","lambda0.1","lambda1", "lambda10","lambda100"]#,"lambda1000"]
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=click_models,
                                  methods=["Fair-I-IPS-LP", "Fair-I-IPS"], iterations=iterations,
                                  plot_individual_fairness=False, multiple_items=multiple_items)


    PLOT_PREFIX = "plots/SynteticOverview/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["Naive", "IPS","Fair-I-IPS", "Fair-E-IPS"], iterations=iterations,
                                  plot_individual_fairness=False)

    items = [Item(i) for i in np.linspace(-1, 1, 20)]

    PLOT_PREFIX = "plots/BiasesVsIPS/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["Naive", "IPS"], iterations=iterations,
                                  plot_individual_fairness=False)

    items = [ Joke(i) for i in np.arange(0,90)]
    popularity = np.zeros(len(items))
    trials = 10
    iterations = 14000

    PLOT_PREFIX = "plots/Exposure/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=[ "Naive", "IPS", "Fair-E-IPS"], iterations=iterations, plot_individual_fairness = False)

    PLOT_PREFIX = "plots/ExposurePers/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["Naive", "Pers", "Fair-E-Pers"],
                                  iterations=iterations, plot_individual_fairness=False)



    PLOT_PREFIX = "plots/Skyline/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["Skyline-Pers"], iterations=iterations, plot_individual_fairness = False)

    load Skyline[0],  Exposure [0,1](Naive Pop, IPS) ExposurePers[1] (Pers) For NDCG Convergence

    iterations = 7000
    PLOT_PREFIX = "plots/ImpactPers/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["Fair-I-IPS",  "Fair-I-Pers"], iterations=iterations, plot_individual_fairness = False)


    """

    """

    PLOT_PREFIX = "plots/Impact/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)

    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                             methods=["Naive", "IPS", "Fair-I-IPS", "Fair-I-Pers"], iterations=iterations, plot_individual_fairness = False)


    PLOT_PREFIX = "plots/NeuralComparison/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["Naive", "IPS", "Pers", "Skyline-Pers"], iterations=iterations,
                                  #methods=["IPS"], iterations=5000,  #
                                  plot_individual_fairness=False)


    """
