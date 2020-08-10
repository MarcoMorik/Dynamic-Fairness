"""
Controlling Fairness and Bias in Dynamic Learning-to-Rank
#TODO Readme
author: Marco Morik
"""
from config import ex
import data_utils
import warnings; warnings.simplefilter('ignore') ##Ignores Warnings for nicer Plots. Disable for Debugging
import birkhoff
from Experiments import *
import plotting
plotting.init_plotting()
import Simulation

birkhoff.TOLERANCE = 10**(-8)

@ex.automain
def __main__(EXPERIMENT, MOVIE_RATING_FILE, PLOT_PREFIX, trials, iterations):
    EXPERIMENT = int(EXPERIMENT)
    if EXPERIMENT == 1:
        multiple_items = [load_news_items(n=30, completly_random=True) for i in range(trials)]
        items = multiple_items[0]
        popularity = np.ones(len(items))
        G = assign_groups(items)
        if not os.path.exists(PLOT_PREFIX):
            os.makedirs(PLOT_PREFIX)
        collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                      methods=["Naive", "IPS", "Fair-I-IPS"], iterations=iterations, multiple_items=multiple_items)

    elif EXPERIMENT == 2:
        multiple_items = [load_news_items(n=30, completly_random=True) for i in range(trials)]
        items = multiple_items[0]
        popularity = np.ones(len(items))
        G = assign_groups(items)
        if not os.path.exists(PLOT_PREFIX):
            os.makedirs(PLOT_PREFIX)
        collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                      methods=["Naive", "IPS"], iterations=iterations, multiple_items=multiple_items)
    elif EXPERIMENT == 3:
        experiment_different_starts(False, trials, iterations, PLOT_PREFIX)
    elif EXPERIMENT == 4:
        compare_controller_LP(trials=trials, iterations=iterations)
    elif EXPERIMENT == 5:
        test_different_groups(trials, iterations, False, prefix=PLOT_PREFIX)
    elif EXPERIMENT == 6:
        test_different_population(trials, iterations, False,prefix=PLOT_PREFIX)
    elif EXPERIMENT == 7:
        data_utils.load_movie_data(movie_ranking_sample_file = MOVIE_RATING_FILE.replace("trial0.npy","trial"))
        movie_experiment(PLOT_PREFIX,
                         ["Naive", "IPS", "Pers", "Skyline-Pers"],
                         MOVIE_RATING_FILE,
                         trials=trials, iterations=iterations, binary_rel=True)
    elif EXPERIMENT == 8:
        data_utils.load_movie_data(movie_ranking_sample_file=MOVIE_RATING_FILE.replace("trial0.npy", "trial"))
        movie_experiment(PLOT_PREFIX,
                         ["Naive", "IPS", "Pers", "Fair-E-Pers"],
                         MOVIE_RATING_FILE,
                         trials=trials, iterations=iterations, binary_rel=True)
    elif EXPERIMENT == 9:
        data_utils.load_movie_data(movie_ranking_sample_file=MOVIE_RATING_FILE.replace("trial0.npy", "trial"))
        movie_experiment(PLOT_PREFIX,
                         ["Naive", "IPS", "Pers", "Fair-I-Pers"],
                         MOVIE_RATING_FILE,
                         trials=trials, iterations=iterations, binary_rel=True)
    elif EXPERIMENT == 10:
        data_utils.load_movie_data(movie_ranking_sample_file=MOVIE_RATING_FILE.replace("trial0.npy", "trial"))
        movie_experiment(PLOT_PREFIX,
                         ["Pers", "Fair-I-Pers",  "Fair-E-Pers"],
                         MOVIE_RATING_FILE,
                         trials=trials, iterations=iterations, binary_rel=True)
