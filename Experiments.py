import os
import numpy as np
from data_utils import *
from Simulation import assign_groups, simulate, collect_relevance_convergence
import matplotlib.pyplot as plt
from plotting import *
from Documents import Item, Movie
def experiment_different_starts(load=False, trials = 50, iterations= 3000, prefix = "", news_items = True):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    methods = ["Naive", "IPS", "Fair-I-IPS"]
    click_model = "PBM_log"
    #starts = [0, 1, 2, 3, 4, 5, 7, 10, 15,25,35]
    starts = [0, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400]
    if news_items:
        multi_items = [load_news_items(completly_random=True) for i in range(trials)]
        multi_G = [assign_groups(x) for x in multi_items]
        items = multi_items[0]
    else:
        items = [Item(i) for i in np.linspace(-1, 1, 20)]
        multi_items = None
    popularity = np.zeros(len(items))
    G = assign_groups(items)

    if not load:
        results = np.zeros((len(methods), len(starts), len(G)))
        results3 = np.zeros((len(methods), len(starts), len(G)))
        resultstd3 =  np.zeros((len(methods), len(starts), len(G)))
        resultstd =  np.zeros((len(methods), len(starts), len(G)))
        overall_fairness = np.zeros((len(methods), len(starts),trials, 4))
        for m, method in enumerate(methods):
            for j, s in enumerate(starts):
                print("Progress {0:.2f}%".format((m*len(starts) + j)/(len(methods)*len(starts))*100))
                top3 = np.zeros(len(G))
                top = np.zeros(len(G))
                std3 =np.zeros((trials,len(G)))
                std = np.zeros((trials, len(G)))
                for i in range(trials):
                    if multi_items is not None:
                        items = multi_items[i]
                        G = multi_G[i]
                    else:
                        np.random.shuffle(items)
                    popularity = np.zeros(len(items))
                    #popularity[G[0]] = s
                    iterations, ranking_hist, popularity_hist, final_ranking, users, ideal, mean_relevances, w_pophist, errors, mean_exposure, fairness_hist, p_pophist = \
                        simulate(popularity, items, ranking_method=method, click_model=click_model, iterations=iterations, head_start=s)

                    for r in final_ranking[:3]:
                        for g in range(len(G)):
                            top3[g] += 1 if r in G[g] else 0
                            std3[i, g] += 1 if r in G[g] else 0
                    for g in range(len(G)):
                        top[g] += 1 if final_ranking[0] in G[g] else 0
                        std[i, g] += 1 if final_ranking[0] in G[g] else 0

                    impact_unfairness = fairness_hist["clicks"][-1,:]
                    for a,b in [(0,1)]:
                        overall_fairness[m, j, i, 0] += np.abs(
                            fairness_hist["prop"][-1, a] / fairness_hist["rel"][ -1, a] - fairness_hist["prop"][ -1, b] / fairness_hist["rel"][-1, b])
                        overall_fairness[m, j, i, 1] += np.abs(
                            fairness_hist["prop"][-1, a] / fairness_hist["true_rel"][ -1, a] - fairness_hist["prop"][ -1, b] / fairness_hist["true_rel"][-1, b])
                        overall_fairness[m, j, i, 2] += np.abs(
                            (fairness_hist["clicks"][-1, a] ) / fairness_hist["rel"][-1, a] - (
                                        fairness_hist["clicks"][-1, b] ) / fairness_hist["rel"][-1, b])
                            #(fairness_hist["clicks"][-1, a]- s) / fairness_hist["rel"][-1, a] -(fairness_hist["clicks"][-1, b]-s) / fairness_hist["rel"][-1, b])
                        overall_fairness[m, j, i, 3] += np.abs(
                            (fairness_hist["clicks"][-1, a] ) / fairness_hist["true_rel"][-1, a] - (
                                        fairness_hist["clicks"][-1, b]) / fairness_hist["true_rel"][-1, b])
                            #(fairness_hist["clicks"][-1, a]-s) / fairness_hist["true_rel"][-1, a] - (fairness_hist["clicks"][-1, b]-s) / fairness_hist["true_rel"][-1, b])

                results[m,j,:] = top / trials
                results3[m,j,:] = top3 / trials
                resultstd[m,j,:] = np.std(std,axis=0)
                resultstd3[m, j, :] = np.std(std, axis=0)


        print("Results Top1", results)
        print("Results Top3", results3)
        print("Fairness: ", overall_fairness)
        np.save(prefix + "top1.npy", results)
        np.save(prefix + "top3.npy", results3)
        np.save(prefix + "top1std.npy", resultstd)
        np.save(prefix + "top3std.npy", resultstd3)
        np.save(prefix + "different_starts_complete.npy",[results,results3,resultstd,resultstd3])
        np.save(prefix + "different_starts_fairness.npy", overall_fairness)
    else:
        results, results3, resultstd, resultstd3 = np.load(prefix + "different_starts_complete.npy")
        overall_fairness = np.load(prefix + "different_starts_fairness.npy")

        #results = np.load("top1.npy")
        #results3 = np.load("top3.npy")
    #print(resultstd)
    methods = ["Naive", "D-ULTR(Glob)", "FairCo(Imp)"]
    for i in range(len(methods)):
        plt.errorbar(starts, results[i,:,0], resultstd[i,:,0], label=methods[i])
    plt.plot(starts,[0.5] * len(starts), label="Fair Outcome", linestyle=":", color ="black")
    plt.xlim([starts[0],starts[-1]])
    plt.ylim(0, 1)
    plt.xlabel("s, Number of clicks in the beginning")
    plt.ylabel("Avg. presence in the top position")
    plt.legend(loc='lower right')
    plt.savefig(prefix + "Richgetricher.pdf", bbox_inches="tight", dpi=600)
    plt.close("all")

    for i in range(len(methods)):
        plt.errorbar(starts, results3[i,:,0], resultstd3[i,:,0], label=methods[i])
    plt.plot(starts,[1.5] * len(starts), label="Fair Outcome", linestyle=":", color ="black")
    plt.ylim(0, 3)
    plt.xlim([starts[0], starts[-1]])
    plt.xlabel("s, Number of clicks in the beginning")
    plt.ylabel("Avg. presence in the top 3 position")
    plt.legend(loc='lower right')
    plt.savefig(prefix + "Richgetricher3.pdf", bbox_inches="tight", dpi=600)

    #### New fairness metric
    plt.close("all")
    for i in range(len(methods)):
        plt.errorbar(starts, np.mean(overall_fairness[i,:,:,3],axis=1), np.std(overall_fairness[i,:,:,3],axis=1), label=methods[i])
    plt.ylim(0, np.max(overall_fairness[:,:,:,3]))
    plt.xlim([starts[0], starts[-1]])
    plt.xlabel("Number of right-leaning users in the beginning")
    plt.ylabel(r'Impact Unfairness ')
    plt.legend(loc='best')
    plt.savefig(prefix + "RichgetricherImpact.pdf", bbox_inches="tight", dpi=600)

    plt.close("all")
    for i in range(len(methods)):
        plt.errorbar(starts, np.mean(overall_fairness[i,:,:,1],axis=1), np.std(overall_fairness[i,:,:,1],axis=1), label=methods[i])
    plt.ylim(0, np.max(overall_fairness[:, :, :, 1]))
    plt.xlim([starts[0], starts[-1]])
    #plt.xlabel("Number of clicks in the beginning")
    plt.xlabel("Number of right-leaning users in the beginning")
    plt.ylabel(r'Exposure Unfairness ')
    plt.legend(loc='best')
    plt.savefig(prefix + "RichgetricherExposure.pdf", bbox_inches="tight", dpi=600)




def test_different_population(trials = 10, iterations = 2000, load = False, prefix = "", news_items=True):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    methods = ["Naive", "IPS", "Fair-I-IPS"]
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #betas = np.linspace(0.05,0.5,8)

    if load:
        all_stats = np.load(prefix + "different_betas2.npy")
    else:
        if news_items:
            multi_items = [load_news_items(completly_random=True) for i in range(trials)]
            items = multi_items[0]
        else:

            items = [Item(i) for i in np.linspace(-1, 1, 20)]
        all_stats = []
        all_users = []
        for b in betas:
            sample_user = lambda: sample_user_base(distribution="bimodal", BI_LEFT=b)
            #get_numerical_relevances = lambda x: get_numerical_relevances_base(x, beta=b)
            stats, users = run_and_save_final_stats(methods, items, trials, iterations, get_users=True, multi_items=multi_items)
            all_stats.append(stats)
            all_users.append(users)
            users = users > 0
            print(np.mean(users, axis=2))

        all_stats = np.asarray(all_stats)
        np.save(prefix + "different_betas2.npy", all_stats)
        np.save(prefix + "different_betasusers.npy", all_users)

    if news_items:
        user_proportion = betas
    else:
        user_proportion = [scipy.stats.beta.cdf(0.5,0.5,x) for x in betas]
    methods = ["Naive", "D-ULTR(Glob)", "FairCo(Imp)"]
    plot_with_errorbar(user_proportion, all_stats, methods, prefix + "different_betas.pdf", "Proportion of Left-Leaning Users")
    plot_with_errorbar(user_proportion, all_stats, methods, prefix + "different_betasExposure.pdf",
                       "Proportion of Left-Leaning Users", impact= False)

    sample_user = lambda: sample_user_base(distribution="bimodal", BI_LEFT=0.5)

def test_different_groups(trials = 10, iterations = 2000, load = False, prefix="plots/"):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    methods = ["Naive", "IPS", "Fair-I-IPS"]
    if load:
        all_stats = np.load(prefix + "different_groupsize.npy")
    else:
        item_list = [ [ load_news_items(n_left=i) for _ in range(trials)] for i in range(1,16)]
        #item_list =[ [ Item(-1.* i / ((21.-x))) for i in range(1,21-x)] + [ Item(i / (x+1.)) for i in range(1,x+1)]  for x in range(1,20)]
        print(np.shape(item_list), "should state 15,10,30")
        all_stats = []
        for multi_items in item_list:
            stats = run_and_save_final_stats(methods, multi_items[0], trials, iterations, multi_items=multi_items)
            all_stats.append(stats)

        all_stats = np.asarray(all_stats)
        np.save(prefix + "different_groupsize.npy", all_stats)
    fraction = [i / 30. for i in range(1,16)]
    methods = ["Naive", "D-ULTR(Glob)", "FairCo(Imp)"]
    plot_with_errorbar(fraction, all_stats, methods, prefix + "different_groupsize.pdf", "Proportion of Left-Leaning Items")
    plot_with_errorbar(fraction, all_stats, methods, prefix + "different_groupsizeExposure.pdf", "Proportion of Left-Leaning Items", impact=False)


def run_and_save_final_stats(methods, items, trials=10, iterations=2000, get_users=False, multi_items=None):
    click_model = "PBM_log"
    final_stats = np.zeros((len(methods), trials, 3))
    run_data = []
    if get_users:
        user_distribution = np.zeros((len(methods), trials, iterations))
    for m, method in enumerate(methods):
        for i in range(trials):

            print("Progress {0:.2f}%".format((m * trials + i) / (len(methods) * trials) * 100))
            if multi_items is None:
                np.random.shuffle(items)
                G = assign_groups(items)
            else:
                items = multi_items[i]
                G = assign_groups(items)
            pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]
            popularity = np.ones(len(items))
            iterations, ranking_hist, popularity_hist, final_ranking, users, ideal, mean_relevances, w_pophist, errors, mean_exposure, fairness_hist, p_pophist = \
                simulate(popularity, items, ranking_method=method, click_model=click_model, iterations=iterations)

            final_stats[m, i, 0] = np.mean(fairness_hist["NDCG"])
            if get_users:
                user_distribution[m, i, :] = np.asarray([u[0] for u in users]).flatten()
            for a, b in pair_group_combinations:
                final_stats[m, i, 1] += np.abs(fairness_hist["prop"][-1, a] / fairness_hist["true_rel"][-1, a] -
                                               fairness_hist["prop"][-1, b] / fairness_hist["true_rel"][-1, b])

                final_stats[m, i, 2] += np.abs(fairness_hist["clicks"][-1, a] / fairness_hist["true_rel"][-1, a] -
                                               fairness_hist["clicks"][-1, b] / fairness_hist["true_rel"][-1, b])

    final_stats[:, :, 1:] /= len(pair_group_combinations)
    if get_users:
        return final_stats, user_distribution
    return final_stats

@ex.capture
def compare_controller_LP(PLOT_PREFIX, trials = 20, iterations = 3000):

    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    multiple_items = [load_news_items(n=30, completly_random=True) for i in range(trials)]
    items = multiple_items[0]
    popularity = np.ones(len(items))
    G = assign_groups(items)
    click_models = ["lambda0","lambda0.0001", "lambda0.001", "lambda0.01","lambda0.1","lambda1", "lambda10","lambda100"]
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=click_models,
                                  methods=["Fair-I-IPS-LP", "Fair-I-IPS"], iterations=iterations,
                                  plot_individual_fairness=False, multiple_items=multiple_items)
    load_and_plot_lambda_comparison(PLOT_PREFIX, trials)


def news_experiment():
    items = load_news_items()
    popularity = np.ones(len(items))
    G = assign_groups(items)
    trials = 100
    iterations = 3000
    """
    #PLOT_PREFIX = "News_data/plots/BiasesVsIPSDiffNoise/"
    PLOT_PREFIX = "News_data/plots/BiasesVsIPS/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["Naive", "IPS"], iterations=iterations,
                                  plot_individual_fairness=True)

    #PLOT_PREFIX = "News_data/plots/SynteticOverviewDiffNoise/"
    PLOT_PREFIX = "News_data/plots/SynteticOverview/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["Naive", "IPS", "Fair-I-IPS", "Fair-E-IPS"], iterations=iterations,
                                  plot_individual_fairness=True)


    PLOT_PREFIX = "News_data/plots/ImpactPers/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["Fair-I-IPS", "Fair-I-Pers"], iterations=iterations,
                                  plot_individual_fairness=False)

    PLOT_PREFIX = "News_data/plots/NeuralComparison/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["Naive", "IPS", "Pers", "Skyline-Pers"], iterations=iterations,
                                  plot_individual_fairness=False)

    PLOT_PREFIX = "News_data/plots/ImpactPersMoreLeft/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["IPS", "Pers", "Fair-I-IPS", "Fair-I-Pers"], iterations=iterations,
                                  plot_individual_fairness=False)

    """
    PLOT_PREFIX = "News_data/plots/Rel_Convergence_final/"
    multiple_items = [load_news_items(n=30, completly_random=True) for i in range(trials)]
    items = multiple_items[0]
    popularity = np.ones(len(items))
    G = assign_groups(items)
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["Naive", "IPS"], iterations=iterations,
                                  plot_individual_fairness=True, multiple_items=multiple_items)
    return

    PLOT_PREFIX = "News_data/plots/Random_data_final_unconditionalNDCG/"
    multiple_items = [load_news_items(n=30, completly_random=True) for i in range(trials)]
    items = multiple_items[0]
    popularity = np.ones(len(items))
    G = assign_groups(items)
    print("Group proportion", len(G[0]), len(G[1]))
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  # methods=["Naive", "IPS", "Pers", "Fair-I-IPS", "Fair-I-Pers", "Fair-E-IPS", "Fair-E-Pers"], iterations=iterations,
                                  methods=["Naive", "IPS", "Fair-I-IPS", "Fair-E-IPS"], iterations=iterations,
                                  plot_individual_fairness=True, multiple_items=multiple_items)


@ex.capture
def movie_experiment(PLOT_PREFIX, methods, MOVIE_RATING_FILE, trials=4, iterations=5000, binary_rel=False):


    _, _, groups = load_movie_data_saved(MOVIE_RATING_FILE)
    items = []
    for i, g in enumerate(groups):
        items.append(Movie(i, g))
    popularity = np.ones(len(items))
    G = assign_groups(items)

    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)

    if binary_rel:
        multi = -1
    else:
        multi = None
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],  # ["PBM_log"],
                                  methods=methods, iterations=iterations,
                                  plot_individual_fairness=True, multiple_items=multi)
