from config import ex
import numpy as np
from scipy.stats import truncnorm
import scipy.integrate
import scipy.stats
import random
import pandas as pd
import time
import warnings; warnings.simplefilter('ignore') ##Ignores Warnings for nicer Plots. Disable for Debugging
import data_utils
import os
import birkhoff
import relevance_network
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotting import *

from Documents import Item, Movie
"""##User Affinity and Distribution"""

def assign_groups(items):
    n_groups = max([i.g for i in items])+1
    G = [ [] for i in range(n_groups)]
    for i, item in enumerate(items):
        G[item.g].append(i)
    return G

#################### Calculate Score between Items and Use #################

#Funktions for User score, position score, assigning groups and  User distributions
@ex.capture
def affinity_score(user, items, bernulli = True, DATA_SET=1):
    if DATA_SET == 1:
        if (type(items) == list):
            return np.asarray([user[0][x.id] for x in items])
        else:
            return user[0][items.id]
    elif DATA_SET == 0:
        #User normal distribution pdf without the normalization factor
        if(type(items) == list):
            item_affs = np.asarray([x.p for x in items])
            item_quality = np.asarray([x.q for x in items])
        else:
            item_affs = items.p
            item_quality = items.q

        #Calculating the Affnity Probability for each Item, based user polarity and user Openness
        aff_prob = np.exp(-(item_affs - user[0])**2 / (2*user[1]**2))*item_quality

        # Binarize The probability of Relevance to Actual relevance of an User
        aff_prob = np.random.rand(*np.shape(aff_prob)) < aff_prob
        return aff_prob
    
    
####### Calculate The position Bias for each of the N Positions #########
def position_bias(n, model="PBM_log", ranked_relevances = None):
    if(model=="PBM_log"):#Position Based Model with inverse log(rank)
        pos = 1/(np.log2(2+np.arange(n)))
        pos /= np.max(pos)
    elif(model=="PBM_inv"):#Position Based Model with inverse rank
        scale = 1
        pos = (1/(1+ np.arange(n)))**scale
    elif(model=="Cascade" or model=="DCM"):
        assert(ranked_relevances is not None)
        if(model=="Cascade"): #Cascade Model
            gamma_click = 0
            gamma_no = 1
        else: #Dependent Click Model
            gamma_click = 0.5
            gamma_no = 0.9
        if(np.max(ranked_relevances) >1):
            ranked_relevances = ranked_relevances / np.max(ranked_relevances)
        pos = np.ones(n)
        for i in range(1, len(pos)):
            pos[i] = pos[i-1]* (gamma_click * ranked_relevances[i-1]+ gamma_no* (1-ranked_relevances[i-1]))
    elif model == "PBM_TEST":
        pos = np.ones(n)
    else:
        print("Could not find", model)
    return pos


###### Calculate NDCG Score

def get_ndcg_score(ranking, true_relevances, click_model = "PBM_log"):
    dcg = np.sum(true_relevances[ranking] / np.log2(2+np.arange(len(ranking))))
    idcg = np.sum(np.sort(true_relevances)[::-1] / np.log2(2+np.arange(len(ranking))))
    if dcg is None or idcg is None or dcg/idcg is None:
        print("Some kind of None appeard with",dcg, idcg, dcg/idcg)
    if(idcg ==0):
        return 1
    return dcg / idcg

@ex.capture
def get_numerical_relevances(items, DATA_SET, MOVIE_RATING_FILE):
    if DATA_SET == 0:
        users = [data_utils.sample_user_base(distribution="bimodal") for i in range(50000)]
        aff = [affinity_score(u, items) for u in users]
        return np.mean(np.asarray(aff), axis=0)
    elif DATA_SET == 1:
        ranking, _, _ = data_utils.load_movie_data_saved(MOVIE_RATING_FILE)
        return np.mean(ranking, axis=0)  # Mean over all users

#Function to obtain a new users, Depending on the Dataset

class Usersampler:
    @ex.capture
    def __init__(self, DATA_SET, BI_LEFT, MOVIE_RATING_FILE):
        self.data_set = DATA_SET
        if DATA_SET == 1:
            self.sample_user_generator = data_utils.sample_user_movie(MOVIE_RATING_FILE)
        if DATA_SET == 0:
            self.BI_LEFT = BI_LEFT

    def get_user(self):
        if self.data_set == 0:
            return data_utils.sample_user_base(distribution="bimodal", BI_LEFT=self.BI_LEFT)
        elif self.data_set == 1:
            return next(self.sample_user_generator)


def get_ranking(user, popularity, items, weighted_popularity=None, G=None, ranking_method="Naive", click_model="PBM_log",
                cum_exposure=None, decomp=None, new_fair_rank=False, nn=None, integral_fairness=None):
    """
    Get the Ranking and position Bias
    For the Linear Program, we also return the current ranking Decomposition (decomp)
    For Fairness Controlling programs, we also return the Fairness Error (fairess_error)
    """
    n = len(popularity)
    click_prob = np.zeros(n)
    fairness_error = None

    # Ranking of the entries
    if (ranking_method == "Naive"):
        ranking = pop_rank(popularity)
    elif (ranking_method == "IPS"):
        assert (weighted_popularity is not None)
        ranking = IPS_rank(weighted_popularity)
    elif ("IPS-LP" in ranking_method):
        # Try Linear Programm for fair ranking, when this fails, use last ranking
        if new_fair_rank or decomp is None:
            if (ranking_method == "Fair-E-IPS-LP"):
                group_fairness = get_unfairness(cum_exposure, weighted_popularity, G, error=False)
                decomp = fair_rank(items, weighted_popularity, debug=False,
                                   group_click_rel=group_fairness, impact=False)
            elif (ranking_method == "Fair-I-IPS-LP"):
                group_fairness = get_unfairness(popularity, weighted_popularity, G, error=False)
                decomp = fair_rank(items, weighted_popularity, debug=False,
                                   group_click_rel=group_fairness, impact=True)
            else:
                raise Exception("Unknown Fair method specified")
        if decomp is not None:
            p_birkhoff = np.asarray([np.max([0, x[0]]) for x in decomp])
            p_birkhoff /= np.sum(p_birkhoff)
            sampled_r = np.random.choice(range(len(decomp)), 1, p=p_birkhoff)[0]
            ranking = np.argmax(decomp[sampled_r][1], axis=0)
        else:
            ranking = IPS_rank(weighted_popularity)
    elif (ranking_method == "Fair-I-IPS"):
        fairness_error = get_unfairness(popularity, weighted_popularity, G, error=True)
        ranking = controller_rank(weighted_popularity, fairness_error)
    elif (ranking_method == "Fair-E-IPS"):
        fairness_error = get_unfairness(cum_exposure, weighted_popularity, G, error=True)
        ranking = controller_rank(weighted_popularity, fairness_error)
    elif ("Pers" in ranking_method):
        if nn is None:
            ranking = IPS_rank(weighted_popularity)
        elif "Fair-E-Pers" == ranking_method:
            fairness_error = get_unfairness(cum_exposure, weighted_popularity, G, error=True)
            ranking = neural_rank(nn, items, user, e_p=fairness_error)
        elif "Fair-I-Pers" == ranking_method:
            fairness_error = get_unfairness(popularity, weighted_popularity, G, error=True)
            ranking = neural_rank(nn, items, user, e_p=fairness_error)
        else:
            ranking = neural_rank(nn, items, user)
    elif (ranking_method == "Random"):
        ranking = random_rank(weighted_popularity)
    else:
        print("could not find a ranking method called: " + ranking_method)
        raise Exception("No Method specified")

    # create prob of click based on position
    pos = position_bias(n, click_model, weighted_popularity[ranking])

    # reorder position probabilities to match popularity order
    pos_prob = np.zeros(n)
    pos_prob[ranking] = pos
    return pos_prob, ranking, decomp, fairness_error


def get_unfairness(clicks, rel, G, error=False):
    """
    Get the Unfairess
    Input Clicks (Cum_Exposure for Exposure Unfairness, Clicks for Impact Unfairness)
    If Error, we return the difference to the best treated group,
    Otherwise just return the Exposure/Impact per Relevance
    """
    n = len(clicks)
    group_clicks = [sum(clicks[G[i]]) for i in range(len(G))]
    group_rel = [max(0.0001, sum(rel[G[i]])) for i in range(len(G))]
    group_fairness = [group_clicks[i] / group_rel[i] for i in range(len(G))]
    if (error):
        best = np.max(group_fairness)
        fairness_error = np.zeros(n)
        for i in range(len(G)):
            fairness_error[G[i]] = best - group_fairness[i]
        return fairness_error
    else:
        return group_fairness


# simulation function returns number of iterations until convergence
@ex.capture
def simulate(popularity, items, ranking_method="Naive", click_model="PBM_log", iterations=2000,
             numerical_relevance=None, head_start=-1, DATA_SET=0, HIDDEN_UNITS=64, PLOT_PREFIX="", user_generator=None):
    #global sample_user
    """
    :param popularity: Initial Popularity
    :param items:  Items/Documents
    :param ranking_method: Method to Use: eg. Naiva, IPS, Pers, Fair-I
    :param click_model: Clickmodel  (PBM_log)
    :param iterations: Iterations/User to sample
    :param numerical_relevance: Use numerical relevance or sampled
    :return count, hist, pophist, ranking, users, ideal_ranking, mean_relevances, w_pophist, nn_errors, mean_exposure, fairness_hist, p_pophist:
    count: Iterations run
    hist: Ranking History
    pophist: Click_History
    ranking: Final ranking
    users: Users sampled
    ideal_ranking: Optimal Ranking
    mean_relevances: Mean Relevance per Item
    w_pophist: Weighted IPS Rating
    nn_errors: Error of Neural Network
    mean_exposure: Mean Exposure per Item
    fairness_hist: Propensities, clicks, estimated_relevance, true_rel per Group and  NDCG
    p_pophist: Personalized Relevance history
    """
    #Initialize Variables
    G = assign_groups(items)
    weighted_popularity = np.asarray(popularity, dtype=np.float32)
    popularity = np.asarray(popularity)
    pophist = np.zeros((iterations, len(items)))
    w_pophist = np.zeros((iterations, len(items)))
    if "Pers" in ranking_method:
        p_pophist = np.zeros((iterations, len(items)))
    else:
        p_pophist = None
    users = []
    aff_scores = np.zeros((iterations, len(items)))
    relevances = np.zeros(len(items))
    cum_exposure = np.zeros(len(items))
    hist = np.zeros((iterations, len(popularity)))
    decomp = None
    group_prop = np.zeros((iterations, len(G)))
    group_clicks = np.zeros((iterations, len(G)))
    group_rel = np.zeros((iterations, len(G)))
    true_group_rel = np.zeros((iterations, len(G)))
    cum_fairness_error = np.zeros(len(items))
    NDCG = np.zeros(iterations)
    if (numerical_relevance is None):
        numerical_relevance = get_numerical_relevances(items)

    # counters
    count = 0
    nn_errors = np.zeros(iterations)
    nn = None
    if user_generator is None:
        user_generator = Usersampler()
    for i in range(iterations):
        count += 1
        #For the Headstart Experiment, we first choose Right then Left Leaning Users
        if (i <= head_start * 2):
            if i == head_start * 2:
                user_generator = Usersampler(BI_LEFT=0.5)
            elif i < head_start:
                user_generator = Usersampler(BI_LEFT=0)
            else:
                user_generator = Usersampler(BI_LEFT=1)

        # choose user
        user = user_generator.get_user()
        users.append(user)
        aff_probs = affinity_score(user, items)
        relevances += aff_probs

        # clicking probabilities
        propensities, ranking, decomp, fairness_error = get_ranking(user, popularity, items, weighted_popularity / count, G,
                                                                    ranking_method, click_model, cum_exposure, decomp,
                                                                    count % 100 == 9, nn=nn,
                                                                    integral_fairness=cum_fairness_error / count)

        # update popularity
        popularity, weighted_popularity = simulate_click(aff_probs, propensities, popularity, weighted_popularity,
                                                         ranking, click_model)

        # Save History
        aff_scores[i] = aff_probs
        hist[i, :] = ranking
        cum_exposure += propensities
        pophist[i, :] = popularity
        w_pophist[i, :] = weighted_popularity

        # update neural network
        if "Pers" in ranking_method:
            if (i == 99):  # Initialize Neural Network
                if DATA_SET == 0:
                    train_x = np.asarray(users)
                elif DATA_SET == 1:
                    train_x = np.asarray([u[1] for u in users])
                if not "Skyline" in ranking_method:
                    nn = relevance_network.relevance_estimating_network(np.shape(train_x)[1], output_dim=len(items),
                                                                        hidden_units=HIDDEN_UNITS,
                                                                        news=True,
                                                                        logdir=PLOT_PREFIX)
                    train_y = w_pophist[:i + 1] - np.concatenate((np.zeros((1, len(items))), w_pophist[:i]))
                else:
                    # Supervised Baseline
                    nn = relevance_network.relevance_estimating_network(np.shape(train_x)[1], output_dim=len(items),
                                                                        hidden_units=HIDDEN_UNITS,
                                                                        news=True,
                                                                        supervised=True, logdir=PLOT_PREFIX)
                    train_y = aff_scores[:i + 1]
                nn.train(train_x, train_y, epochs=2000, trial=i)
            elif (i > 99 and i % 10 == 9):
                if "Skyline" in ranking_method:
                    train_y = aff_scores[:i + 1]
                else:
                    train_y = np.concatenate((train_y, w_pophist[i - 9:i + 1] - w_pophist[i - 10:i]))
                    # assert(np.array_equal(train_y,w_pophist[:i + 1] - np.concatenate((np.zeros((1, len(items))), w_pophist[:i]))))
                if DATA_SET == 1:
                    train_x = np.concatenate((train_x, np.asarray([u[1] for u in users[-10:]])))
                else:
                    train_x = np.concatenate((train_x, np.asarray([u for u in users[-10:]])))

                nn.train(train_x, train_y, epochs=10, trial=i)

            if DATA_SET and i >= 99:
                predicted_relevances = nn.predict(user[1])
            elif i >= 99:
                predicted_relevances = nn.predict(user)
            if i >= 99:
                nn_errors[i] = np.mean((predicted_relevances - aff_probs) ** 2)
                p_pophist[i, :] = predicted_relevances
            else:
                p_pophist[i, :] = weighted_popularity


        # Save statistics
        if (fairness_error is not None):
            cum_fairness_error += fairness_error

        if DATA_SET:
            NDCG[i] = get_ndcg_score(ranking, user[0])
        else:
            NDCG[i] = get_ndcg_score(ranking, aff_probs)  # numerical_relevance)

        group_prop[i, :] = [np.sum(cum_exposure[G[i]]) for i in range(len(G))]
        group_clicks[i, :] = [np.sum(popularity[G[i]]) for i in range(len(G))]
        if ("Pers" in ranking_method):
            group_rel[i, :] = [np.sum(p_pophist[i, G[g]]) for g in range(len(G))]
        elif ("Naive" in ranking_method):
            group_rel[i, :] = [np.sum(pophist[i, G[g]]) for g in range(len(G))]
        else:
            group_rel[i, :] = [np.sum(weighted_popularity[G[g]]) for g in range(len(G))]

        true_group_rel[i, :] = [np.sum(numerical_relevance[G[g]]) * count for g in range(len(G))]


    ideal_vals, ideal_ranking = ideal_rank(users, items)

    mean_relevances = relevances / count
    mean_exposure = cum_exposure / count

    fairness_hist = {"prop": group_prop, "clicks": group_clicks, "rel": group_rel, "true_rel": true_group_rel,
                     "NDCG": NDCG}
    return count, hist, pophist, ranking, users, ideal_ranking, mean_relevances, w_pophist, nn_errors, mean_exposure, fairness_hist, p_pophist


def simulate_click(aff_probs, propensities, popularity, weighted_popularity, ranking, click_model):
    if "PBM" in click_model:
        rand_var = np.random.rand(len(aff_probs))
        rand_prop = np.random.rand(len(propensities))
        viewed = rand_prop < propensities
        clicks = np.logical_and(rand_var < aff_probs, viewed)
        popularity += clicks
        weighted_popularity += clicks / propensities

    elif click_model == "Cascade" or click_model == "DCM":
        c_stop = 1
        if click_model == "Cascade":
            gamma_click = 0
            gamma_no = 1
        else:
            gamma_click = 0.5
            gamma_no = 0.98
        for i, r in enumerate(ranking):
            if random.random() < aff_probs[r]:
                popularity[r] += 1
                weighted_popularity[r] += 1. / c_stop
                c_stop *= gamma_click
                if random.random() > gamma_click:
                    break
            else:
                if random.random() > gamma_no:
                    break
                c_stop *= gamma_no
    else:
        raise Exception("Could not find the clickmodel")
    return popularity, weighted_popularity


"""##Ranking Functions"""
#Ranking Functions:
#Popularity Ranking
def pop_rank(popularity):
    return np.argsort(popularity)[::-1]

#Inverse Propensity Ranking
def IPS_rank(weighted_popularity):
    return np.argsort(weighted_popularity)[::-1]

#Random Ranking
def random_rank(weighted_popularity):
    ranking = np.arange(len(weighted_popularity))
    np.random.shuffle(ranking)
    return ranking

#Rank using a simple P Controller
@ex.capture
def controller_rank(weighted_popularity, e_p, KP= 0.01):
    return np.argsort(weighted_popularity + KP * e_p )[::-1]

#Ranking with neural network relevances
@ex.capture
def neural_rank(nn, items, user, DATA_SET = 1, e_p = 0, KP= 0.01 ):
    if DATA_SET == 1 :
        x_test = np.asarray(user[1])
    elif DATA_SET == 0:
        x_test = np.asarray(user)
        #x_test = np.asarray(list(map(lambda x: x.get_features(), items)))
    #print("Input  shape", x_test.shape)
    relevances = nn.predict(x_test)
    return np.argsort(relevances+ KP * e_p)[::-1]

#Fair Ranking
@ex.capture
def fair_rank(items, popularity,ind_fair=False, group_fair=True, debug=False, w_fair = 1, group_click_rel = None, impact=True, LP_COMPENSATE_W=10):
    n = len(items)
    pos_bias = position_bias(n)
    G = assign_groups(items)
    n_g, n_i = 0, 0
    if(group_fair):
        n_g += (len(G)-1)*len(G)
    if(ind_fair):
        n_i += n * (n-1)

    n_c = n**2 + n_g + n_i


    c = np.ones(n_c)
    c[:n**2] *= -1
    c[n**2:] *= w_fair
    A_eq = []
    #For each Row
    for i in range(n):
        A_temp = np.zeros(n_c)
        A_temp[i*n:(i+1)*n] = 1
        assert(sum(A_temp)==n)
        A_eq.append(A_temp)
        c[i*n:(i+1)*n] *= popularity[i]

    #For each coloumn
    for i in range(n):
        A_temp = np.zeros(n_c)
        A_temp[i:n**2:n] = 1
        assert(sum(A_temp)==n)
        A_eq.append(A_temp)
        #Optimization
        c[i:n**2:n] *= pos_bias[i]
    b_eq = np.ones(n*2)
    A_eq = np.asarray(A_eq)
    bounds = [(0,1) for _ in range(n**2)] + [(0,None) for _ in range(n_g+n_i)]


    A_ub = []
    b_ub = np.zeros(n_g+n_i)
    if(group_fair):
        U = []
        for group in G:
            #Avoid devision by zero
            u = np.max([sum(np.asarray(popularity)[group]), 0.01])
            U.append(u)
        comparisons = list(permutations(np.arange(len(G)),2))
        j = 0
        for a,b in comparisons:
            f = np.zeros(n_c)
            if len(G[a]) > 0 and len(G[b])>0: # and U[a] >= U[b]: #len(G[a]) * U[a] >= len(G[b]) *U[b]: for comparing mean popularity
                for i in range(n):
                    #tmp1 = 1. / U[a] if i in G[a] else 0
                    #tmp2 = 1. / U[b] if i in G[b] else 0
                    if impact:
                        tmp1 = popularity[i] / U[a] if i in G[a] else 0
                        tmp2 = popularity[i] / U[b] if i in G[b] else 0
                    else:
                        tmp1 = 1. / U[a] if i in G[a] else 0
                        tmp2 = 1. / U[b] if i in G[b] else 0
                    #f[i*n:(i+1)*n] *= max(0, sign*(tmp1 - tmp2))
                    f[i*n:(i+1)*n] =  (tmp1 - tmp2) # * popularity[i] for equal impact instead of equal Exposure
                for i in range(n):
                    f[i:n**2:n] *= pos_bias[i]
                f[n**2+j] = -1
                if group_click_rel is not None:
                    b_ub[j] = LP_COMPENSATE_W * (group_click_rel[b] - group_click_rel[a])
            j += 1
            A_ub.append(f)

    if(ind_fair):
        comparisons = list(permutations(np.arange(len(popularity)),2))
        j = 0
        for a,b in comparisons:
            f = np.zeros(n_c)
            if(popularity[a] >= popularity[b]):
                tmp1 = 1. / np.max([0.01,popularity[a]])
                tmp2 = 1. / np.max([0.01,popularity[b]])
                f[a*n:(a+1)*n] = tmp1
                f[a*n:(a+1)*n] *= pos_bias
                f[b*n:(b+1)*n] = -1 *  tmp2
                f[b*n:(b+1)*n] *= pos_bias

                f[n**2+n_g+j] = -1
            j += 1
            A_ub.append(f)

    res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options=dict(bland =True, tol=1e-12), method = "interior-point")
    probabilistic_ranking = np.reshape(res.x[:n**2],(n,n))
    #probabilistic_ranking = np.reshape(res.x[:n**2],(n,n))


    if(debug):
        print("Shape of the constrains", np.shape(A_eq), "with {} items and {} groups".format(n, len(G)))
        print("Fairness constraint:", np.round(np.dot(A_eq,res.x),4))
        #print("Constructed probabilistic_ranking with score {}: \n".format(res.fun), np.round(probabilistic_ranking,2))
        print("Col sum: ", np.sum(probabilistic_ranking,axis=0))
        print("Row sum: ", np.sum(probabilistic_ranking,axis=1))
        #plt.matshow(A_eq)
        #plt.colorbar()
        #plt.plot()
        plt.matshow(probabilistic_ranking)
        plt.colorbar()
        plt.plot()

    #Sample from probabilistic ranking using Birkhoff-von-Neumann decomposition
    try:
        decomp = birkhoff.birkhoff_von_neumann_decomposition(probabilistic_ranking)
    except:
        decomp = birkhoff.approx_birkhoff_von_neumann_decomposition(probabilistic_ranking)

        if debug:
            print("Could get a approx decomposition with {}% accuracy".format(100*sum([x[0] for x in decomp])) )
            #print(probabilistic_ranking)

    return decomp


def ideal_rank(users, item_affs):
    aff_prob = np.zeros(len(item_affs))
    for user in users:
        aff_prob += affinity_score(user, item_affs)

    return aff_prob, (np.argsort(aff_prob)[::-1])





# Function that simulates and monitor the convergence to the relevance + the developement of cummulative fairness
@ex.capture
def collect_relevance_convergence(items, start_popularity, trials=10, methods=["Naive", "IPS"],
                                  click_models=["PBM_log"], iterations=2000, plot_individual_fairness=True,
                                  multiple_items=None, PLOT_PREFIX="", MOVIE_RATING_FILE=""):

    global get_numerical_relevances

    rel_diff = []
    if multiple_items is None:
        G = assign_groups(items)
    else:
        if multiple_items == -1:
            G = assign_groups(items)
        else:
            assert (len(multiple_items) == trials)
            G = assign_groups(multiple_items[0])
    overall_fairness = np.zeros((len(click_models) * len(methods), trials, iterations, 4))
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]
    count = 0
    run_data = []
    frac_c = [[] for i in range(len(G))]
    nn_errors = []
    method_dict = {"Naive": "Naive", "IPS": r'$\hat{R}^{IPS}(d)$', "Pers": "D-ULTR", "Skyline-Pers": "Skyline",
                   "Fair-I-IPS": "FairCo(Imp)", "Fair-E-IPS": "FairCo(Exp)", "Fair-I-Pers": "FairCo(Imp)",
                   "Fair-E-Pers": "FairCo(Exp)", "Fair-I-IPS-LP": "LinProg(Imp)", "Fair-E-IPS-LP": "LinProg(Exp)"}
    user_generator = None
    for click_model in click_models:

        if "lambda" in click_model: #For vcomparing different Lambdas,
            lam = float(click_model.replace("lambda", ""))
            ex.add_config({
                'KP': lam,
                'W_FAIR': lam
            })
            click_model = "PBM_log"
        for method in methods:
            start_time = time.time()
            rel_diff_trial = []
            # rel_diff_top20 = []
            fairness = {"prop": np.zeros((trials, iterations, len(G))),
                        "clicks": np.zeros((trials, iterations, len(G))), "rel": np.zeros((trials, iterations, len(G))),
                        "true_rel": np.zeros((trials, iterations, len(G))), "NDCG": np.zeros((trials, iterations))}
            nn_error_trial = []
            for i in range(trials):
                if multiple_items is not None:
                    if multiple_items == -1:  # Load a new bernully relevance table
                        MOVIE_RATING_FILE = MOVIE_RATING_FILE.replace("trial{}.npy".format(i-1),"trial{}.npy".format(i))
                        #MOVIE_RATING_FILE = "data/movie_data_binary_latent_5Comp_trial{}.npy".format(i)
                        user_generator = Usersampler(MOVIE_RATING_FILE=MOVIE_RATING_FILE)
                        ranking, _, _ = data_utils.load_movie_data_saved(MOVIE_RATING_FILE)
                        get_numerical_relevances = lambda x: np.mean(ranking, axis=0)

                    else:
                        items = multiple_items[i]
                        G = assign_groups(items)
                popularity = np.copy(start_popularity)
                # Run Simulation
                iterations, ranking_hist, popularity_hist, final_ranking, users, ideal, mean_relevances, w_pophist, errors, mean_exposure, fairness_hist, p_pophist = \
                    simulate(popularity, items, ranking_method=method, click_model=click_model, iterations=iterations, user_generator=user_generator)
                ranking_hist = ranking_hist.astype(int)
                if "Pers" in method:
                    nn_error_trial.append(errors)

                # Calculate the relevance difference between true relevance and approximation
                # Diff = |rel' - rel|
                if method == "Naive":
                    rel_estimate = popularity_hist / np.arange(1, iterations + 1)[:, np.newaxis]
                elif "Pers" in method:
                    p_pophist[99:, :] = [np.sum(p_pophist[98:100 + i, :], axis=0) for i in range(len(p_pophist) - 99)]

                    rel_estimate = p_pophist / (np.arange(iterations) + 1)[:, np.newaxis]
                else:
                    rel_estimate = w_pophist / np.arange(1, iterations + 1)[:, np.newaxis]

                rel_diff_trial.append(np.mean(np.abs(rel_estimate - (mean_relevances)[np.newaxis, :]), axis=1))
                # if len(items) > 20 and False:
                #        rel_top20 = np.mean(np.abs(rel_estimate[:,ranking_hist[:,:20]] - mean_relevances[np.newaxis,ranking_hist[:,:20]]),axis = 1)
                #        rel_diff_top20.append(rel_top20)

                # Cummulative Fairness per Iteration summed over trials
                for key, value in fairness_hist.items():
                    fairness[key][i] = value
                if (trials <= 1):
                    # Plot Group Clicks and Items Average Rank
                    group_item_clicks(popularity_hist[-1], G)
                    plot_average_rank(ranking_hist, G)
                    print("Relevance Difference: ", np.sum((mean_relevances - rel_estimate[-1]) ** 2))

                    # Plot Ranking History
                    plt.title("Ranking History")
                    plt.axis([0, iterations, 0, len(items)])
                    if len(G) <= 3:
                        group_colors = {0: "blue", 1: "red", 2: "black"}
                        group_labels = {0: "Negative", 1: "Positive", 2: "black"}
                    else:
                        group_colors = [None for i in range(len(G))]
                    item_rank_path = np.ones((iterations, len(items)))
                    for i in range(iterations):
                        item_rank_path[i, ranking_hist[i, :]] = np.arange(len(items))
                    for i in range(len(items)):
                        group_color_i = group_colors[[x for x in range(len(G)) if i in G[x]][0]]
                        plt.plot(np.arange(iterations), item_rank_path[:, i], color=group_color_i)

                    custom_lines = [Line2D([0], [0], color="blue", lw=4),
                                    Line2D([0], [0], color="red", lw=4)]

                    plt.legend(custom_lines, ['Negative', 'Positive'])
                    # plt.show()
                    plt.legend()
                    plt.savefig(PLOT_PREFIX + "Rankinghistory_" + click_model + "_" + method + ".pdf",
                                bbox_inches="tight")

            print("Time for " + click_model + " " + method + " was: {0:.4f}".format(time.time() - start_time))

            if "Pers" in method:
                mean_trial_error = np.mean(np.asarray(nn_error_trial), axis=0)
                nn_errors.append(mean_trial_error)

            count += 1
            # Plot the Fairness per Group for a single model
            if (plot_individual_fairness):
                plot_fairness_over_time(fairness, G, method)
            # Collect Data for later
            run_data.append(fairness)

            for i in range(len(G)):
                frac_c[i].append(np.mean(fairness["clicks"][:, -1, i]) / iterations)

            if (len(rel_diff_trial) == 1):
                rel_tmp = np.asarray(rel_diff_trial[0])
                rel_std = np.zeros(np.shape(rel_tmp))
            else:
                rel_tmp = np.mean(np.asarray(rel_diff_trial), axis=0)
                rel_std = np.std(np.asarray(rel_diff_trial), axis=0)
            rel_diff.append([rel_tmp, method_dict[method], rel_std])
            # if(len(items)>20 and False):
            #    rel_diff.append((np.mean(np.asarray(rel_diff_top20),axis=0),click_model.replace("_log","") + " "+ method + "top 20"))

    np.save(PLOT_PREFIX + "Fairness_Data.npy", run_data)
    # Plot NDCG
    plt.figure("NDCG")
    # plt.title("Average NDCG")
    # labels = [ a + "\n" + b for a in click_models for b in methods]
    labels = [b for a in click_models for b in methods]
    for i, nd in enumerate(run_data):
        plot_ndcg(np.mean(nd["NDCG"], axis=0), label=labels[i], plot=False, window_size=100, std=nd["NDCG"])
    plt.legend()
    ax = plt.gca()
    plt.savefig(PLOT_PREFIX + "NDCG.pdf", bbox_inches="tight", dpi=800)
    plt.show()
    plt.close("all")

    # Plot Clicks
    plot_click_bar_plot(frac_c, labels, save=True)

    if True:
        plt.close("all")
        # Plot Convergence of Relevance
        for y in rel_diff:
            p = plt.plot(np.arange(len(y[0])), y[0], label=y[1])
            color = p[-1].get_color()
            plt.fill_between(np.arange(len(y[0])), y[0] - y[2],
                             y[0] + y[2], alpha=0.4, color=color)

        plt.legend(loc="best")
        plt.axis([0, len(y[0]), 0, 0.3])
        # plt.ylabel("Avg diff between \n True & Estimated Relevance  ")
        plt.ylabel(r'average $|\hat{R}(d) - {R}(d)|$')

        plt.xlabel("Users")
        plt.savefig(PLOT_PREFIX + "Relevance_convergence.pdf", bbox_inches="tight")
        plt.show()

    plot_neural_error(nn_errors, [b for a in click_models for b in methods if "Pers" in b])
    # Plot Unfairness over time between different models

    for i, data in enumerate(run_data):

        for a, b in pair_group_combinations:
            overall_fairness[i, :, :, 0] += np.abs(
                data["prop"][:, :, a] / data["rel"][:, :, a] - data["prop"][:, :, b] / data["rel"][:, :, b])
            overall_fairness[i, :, :, 1] += np.abs(
                data["prop"][:, :, a] / data["true_rel"][:, :, a] - data["prop"][:, :, b] / data["true_rel"][:, :, b])
            overall_fairness[i, :, :, 2] += np.abs(
                data["clicks"][:, :, a] / data["rel"][:, :, a] - data["clicks"][:, :, b] / data["rel"][:, :, b])
            overall_fairness[i, :, :, 3] += np.abs(
                data["clicks"][:, :, a] / data["true_rel"][:, :, a] - data["clicks"][:, :, b] / data["true_rel"][:, :,
                                                                                                b])

    overall_fairness /= len(pair_group_combinations)
    plot_unfairness_over_time(overall_fairness, click_models, methods, True)

    ndcg_full = []
    for data in run_data:
        ndcg_full.append(data["NDCG"])
    plt.close('all')
    combine_and_plot_ndcg_unfairness(ndcg_full,overall_fairness[:, :, :, 1],labels= labels, selection=np.arange(len(run_data)), name=PLOT_PREFIX + "NDCG_UnfairExposure.pdf",type = 0 )
    combine_and_plot_ndcg_unfairness(ndcg_full,overall_fairness[:, :, :, 3],labels= labels, selection=np.arange(len(run_data)), name=PLOT_PREFIX + "NDCG_UnfairImpact.pdf",type = 1 )