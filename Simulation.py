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
from ranking_functions import *
from itertools import permutations

from Documents import Item, Movie, Joke
"""##User Affinity and Distribution"""

def  binominal_dist(m1 = 0.5, std1 = 0.5, m2 = -0.5, std2 = 0.5, BI_LEFT = 0.5):
    """

    """
    if np.random.rand() < BI_LEFT:
        user = truncnorm.rvs(-1, 1, m1, std1, 1)
    else:
        user = truncnorm.rvs(-1, 1, m2, std2, 1)
    # std = u_std
    std = np.random.rand() / 2 + 0.05
    return np.asarray([user,std])

def assign_groups(items):
    n_groups = max([i.g for i in items])+1
    G = [ [] for i in range(n_groups)]
    for i, item in enumerate(items):
        G[item.g].append(i)
    return G

#################### Calculate Score between Items and Use #################

#Funktions for User score, position score, assigning groups and  User distributions
def affinity_score(user, items):
    return(affinity_score_adv(user,items, bernulli=False))

@ex.capture
def affinity_score_adv(user, items, bernulli = True, DATA_SET=1):
    if DATA_SET == 1 or DATA_SET == 2:
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
        aff_prob = np.exp(-(item_affs - user[0])**2 / (2*user[1]**2))*item_quality

        #TODO Changed now to drawing from Bernulli
        if bernulli:
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
            #pos[i] = 1 - p_stop * np.dot(pos[:i], ranked_relevances[:i])
    elif model == "PBM_TEST":
        pos = np.ones(n)
    else:
        print("Could not find", model)
    return pos


###### Calculate NDCG Score

def get_ndcg_score(ranking, true_relevances, click_model = "PBM_log"):
    dcg = np.sum(true_relevances[ranking] / np.log2(2+np.arange(len(ranking))))
    idcg = np.sum(np.sort(true_relevances)[::-1] / np.log2(2+np.arange(len(ranking))))
    #dcg = np.sum(true_relevances[ranking] /position_bias(len(ranking),click_model, true_relevances[ranking]))
    #idcg_rank = np.argsort(true_relevances)[::-1]
    #idcg =  np.sum(idcg_rank / position_bias(len(ranking),click_model,true_relevances[idcg_rank]))
    if dcg is None or idcg is None or dcg/idcg is None:
        print("Some kind of None appeard with",dcg, idcg, dcg/idcg)
    if(idcg ==0):
        return 1
    return dcg / idcg

@ex.capture
def get_numerical_relevances(items, DATA_SET, MOVIE_RATING_FILE):
    if DATA_SET == 0:
        users = [sample_user() for i in range(50000)]
        aff = [affinity_score(u, items) for u in users]
        return np.mean(np.asarray(aff), axis=0)
    elif DATA_SET == 1:
        df, _ = data_utils.load_data()
        relevances = df.mean().as_matrix()
        return np.asarray(relevances)
    elif DATA_SET == 2:
    
        ranking, _, _ = data_utils.load_movie_data_saved(MOVIE_RATING_FILE)
        return np.mean(ranking, axis=0)  # Mean over all users


sample_user = data_utils.get_user_generator()
get_numerical_relevances = get_numerical_relevances()


def click(user, popularity, items, weighted_popularity=None, G=None, ranking_method="Naive", click_model="PBM_log",
          cum_exposure=None, decomp=None, new_fair_rank=False, nn=None, integral_fairness=None):
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
                decomp = fair_rank(items, weighted_popularity, debug=False, w_fair=W_FAIR,
                                   group_click_rel=group_fairness, impact=False)
            elif (ranking_method == "Fair-I-IPS-LP"):
                group_fairness = get_unfairness(popularity, weighted_popularity, G, error=False)
                decomp = fair_rank(items, weighted_popularity, debug=False, w_fair=W_FAIR,
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


def get_unfairness(clicks, rel, G, error=False, boost=False, group_sum=False):
    n = len(clicks)
    group_clicks = [sum(clicks[G[i]]) for i in range(len(G))]
    group_rel = [max(0.0001, sum(rel[G[i]])) for i in range(len(G))]
    group_fairness = [group_clicks[i] / group_rel[i] for i in range(len(G))]

    if (error and not boost):
        best = np.max(group_fairness)
        fairness_error = np.zeros(n)
        for i in range(len(G)):
            fairness_error[G[i]] = best - group_fairness[i]
        return fairness_error
    elif (error and boost):
        best = np.max(group_fairness)
        fairness_error = np.zeros(n)
        for i in range(len(G)):
            fairness_error[G[i]] = best / group_fairness[i]
        return fairness_error
    else:
        return group_fairness


# simulation function returns number of iterations until convergence
@ex.capture
def simulate(popularity, items, ranking_method="Naive", click_model="PBM_log", iterations=2000,
             numerical_relevance=None, head_start=-1, DATA_SET=0, HIDDEN_UNITS):
    global sample_user
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
    # aff_scores = []
    aff_scores = np.zeros((iterations, len(items)))
    relevances = np.zeros(len(items))
    cum_exposure = np.zeros(len(items))
    hist = np.zeros((iterations, len(popularity)))
    prev = np.argsort(popularity)
    decomp = None
    group_prop = np.zeros((iterations, len(G)))
    group_clicks = np.zeros((iterations, len(G)))
    group_rel = np.zeros((iterations, len(G)))
    true_group_rel = np.zeros((iterations, len(G)))
    cum_fairness_error = np.zeros(len(items))
    NDCG = np.zeros(iterations)
    if (numerical_relevance is None):
        numerical_relevance = get_numerical_relevances(items)
    # params
    threshold = 100

    # counters
    count = 0
    nn_errors = np.zeros(iterations)

    nn = None

    # x_test = np.asarray(list(map(lambda x: x.get_features(), items)))
    for i in range(iterations):
        count += 1

        # choose user
        if (i <= head_start * 2):
            if i == head_start * 2:
                sample_user = lambda: sample_user_base(BI_LEFT=0.5)
            elif i < head_start:
                sample_user = lambda: sample_user_base("bimodal", BI_LEFT=0)
            else:
                sample_user = lambda: sample_user_base("bimodal", BI_LEFT=1)

        user = sample_user()
        users.append(user)
        aff_probs = affinity_score_adv(user, items)
        relevances += aff_probs

        # clicking probabilities
        propensities, ranking, decomp, fairness_error = click(user, popularity, items, weighted_popularity / count, G,
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
                elif DATA_SET == 2 or DATA_SET == 1:
                    train_x = np.asarray([u[1] for u in users])
                if not "Skyline" in ranking_method:
                    nn = relevance_network.relevance_estimating_network(np.shape(train_x)[1], output_dim=len(items),
                                                                        hidden_units=HIDDEN_UNITS, joke=DATA_SET == 1,
                                                                        news=(DATA_SET == 0 or DATA_SET == 2),
                                                                        logdir=PLOT_PREFIX)
                    train_y = w_pophist[:i + 1] - np.concatenate((np.zeros((1, len(items))), w_pophist[:i]))
                else:
                    # Supervised Baseline
                    nn = relevance_network.relevance_estimating_network(np.shape(train_x)[1], output_dim=len(items),
                                                                        hidden_units=HIDDEN_UNITS, joke=DATA_SET == 1,
                                                                        news=(DATA_SET == 0 or DATA_SET == 2),
                                                                        supervised=True, logdir=PLOT_PREFIX)
                    train_y = aff_scores[:i + 1]
                nn.train(train_x, train_y, epochs=2000, trial=i)
            elif (i > 99 and i % 10 == 9):
                if "Skyline" in ranking_method:
                    train_y = aff_scores[:i + 1]
                else:
                    train_y = np.concatenate((train_y, w_pophist[i - 9:i + 1] - w_pophist[i - 10:i]))
                    # assert(np.array_equal(train_y,w_pophist[:i + 1] - np.concatenate((np.zeros((1, len(items))), w_pophist[:i]))))
                if DATA_SET == 1 or DATA_SET == 2:
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
            # NDCG[i] = get_ndcg_score(ranking, relevances/count) #numerical_relevance)
            NDCG[i] = get_ndcg_score(ranking, aff_probs)  # numerical_relevance)
            # NDCG[i] = get_ndcg_score(ranking, numerical_relevance) # TODO now we use the numerical Relevance instead of personalized
        # NDCG[i] = get_ndcg_score(ranking, relevances/count)
        # print( np.sum(np.abs(numerical_relevance - (relevances/count))))

        group_prop[i, :] = [np.sum(cum_exposure[G[i]]) for i in range(len(G))]
        group_clicks[i, :] = [np.sum(popularity[G[i]]) for i in range(len(G))]
        # group_rel[i,:] = [sum(weighted_popularity[G[i]])/count for i in range(len(G))] #Normalizing by Count to have time normalized relevances
        # true_group_rel[i,:] = [sum(relevances[G[i]])/count for i in range(len(G))]
        if ("Pers" in ranking_method):
            group_rel[i, :] = [np.sum(p_pophist[i, G[g]]) for g in range(len(G))]
        elif ("Naive" in ranking_method):
            group_rel[i, :] = [np.sum(pophist[i, G[g]]) for g in range(len(G))]
        else:
            group_rel[i, :] = [np.sum(weighted_popularity[G[g]]) for g in
                               range(len(G))]  # Having the sum of weighted clicks. Nicer Plots for the fairness measure
        # true_group_rel[i,:] = [sum(relevances[G[i]]) for i in range(len(G))]
        # TODO true rel News
        true_group_rel[i, :] = [np.sum(numerical_relevance[G[g]]) * count for g in
                                range(len(G))]  # Try to avoid the spikes in the beginning

        prev = np.copy(ranking)

    # rank by expected rel over users P(rel)
    # calculate "ideal" rank
    # users = list of "affiliation scores"
    ideal_vals, ideal_ranking = ideal_rank(users, items)

    mean_relevances = relevances / count
    mean_exposure = cum_exposure / count

    # true_group_rel[:10,:]=true_group_rel[10,:][np.newaxis,:]
    fairness_hist = {"prop": group_prop, "clicks": group_clicks, "rel": group_rel, "true_rel": true_group_rel,
                     "NDCG": NDCG}
    return count, hist, pophist, ranking, users, ideal_ranking, mean_relevances, w_pophist, nn_errors, mean_exposure, fairness_hist, p_pophist


def simulate_click(aff_probs, propensities, popularity, weighted_popularity, ranking, click_model):
    if "PBM" in click_model:
        """
        #Adding random Click Noise
        rand_var = np.random.rand(len(aff_probs))

        click_noise = 0
        rel_noise = (aff_probs + click_noise)
        clicks = rand_var < ((rel_noise * propensities))

        #noised_relevance = aff_probs + click_noise # Can leave the 0-1 interval
        #Add noise in an unconstraint space and transpose back to 0-1
        #rel_noise = (aff_probs + click_noise)
        # rel_noise = norm.cdf(norm.ppf(aff_probs) + click_noise)
        """
        rand_var = np.random.rand(len(aff_probs))
        rand_prop = np.random.rand(len(propensities))
        # noise_rand_var = np.random.rand(len(aff_probs))
        viewed = rand_prop < propensities
        clicks = np.logical_and(rand_var < aff_probs, viewed)
        # clicks = np.logical_and(np.logical_and(rand_var < aff_probs, viewed), noise_rand_var <= eps_p)
        # un_rel_click = np.logical_and(noise_rand_var < eps_m, viewed)
        # clicks = np.logical_or(clicks, un_rel_click)
        popularity += clicks
        weighted_popularity += clicks / propensities
        # weighted_popularity += (clicks-eps_m)/(eps_p-eps_m) / propensities

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
        print("Could not find the clickmodel")
        assert (1 == 2)
    return popularity, weighted_popularity



"""##Ranking Functions"""
#Ranking Functions:
#Popularity Ranking
def pop_rank(popularity):
    return np.argsort(popularity)[::-1]

#Inverse Propensity Ranking
def IPS_rank(weighted_popularity):
    return np.argsort(weighted_popularity)[::-1]


#Ranking that boost groups underrepresented so far
def boost_rank(weighted_popularity, fairness_boost):
    return np.argsort(weighted_popularity * fairness_boost)[::-1]


#Random Ranking
def random_rank(weighted_popularity):
    ranking = np.arange(len(weighted_popularity))
    np.random.shuffle(ranking)
    return ranking


#Probabilistic Ranking using Gumble Distribution
def gumble_rank(weighted_popularity):
    return np.argsort(np.random.gumbel(loc = weighted_popularity))[::-1]


#Rank using a simple P Controller
def controller_rank(weighted_popularity, e_p):
    return np.argsort(weighted_popularity + KP * e_p )[::-1]



#P-L Ranking using softmax
def probabilistic_rank(weighted_popularity):
    p =np.exp(PROB_W * weighted_popularity)/sum(np.exp(PROB_W *weighted_popularity))
    return np.random.choice(np.arange(len(p)),len(p),replace=False, p=p)


#Ranking with neural network relevances
def neural_rank(nn, items, user, data_set = DATA_SET, e_p = 0 ):
    if data_set == 1 or data_set == 2:
        x_test = np.asarray(user[1])
    elif data_set == 0:
        x_test = np.asarray(user)
        #x_test = np.asarray(list(map(lambda x: x.get_features(), items)))
    #print("Input  shape", x_test.shape)
    relevances = nn.predict(x_test)
    return np.argsort(relevances+ KP * e_p)[::-1]

#Fair Ranking
def fair_rank(items, popularity,ind_fair=False, group_fair=True, debug=False, w_fair = 1, group_click_rel = None, impact=True):
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
    """
    #If not doubly stochastic, try SinkhornKnopp
    if(not ((np.sum(probabilistic_ranking,axis=0) == 1).all() and (np.sum(probabilistic_ranking,axis=1) == 1).all()) ):
        sk = skp.SinkhornKnopp()
        probabilistic_ranking = sk.fit(probabilistic_ranking)
        print("Constraints not fullfilled")
        print("Col sum: ", np.sum(probabilistic_ranking,axis=0))
        print("Row sum: ", np.sum(probabilistic_ranking,axis=1))
    """

    #Sample from probabilistic ranking using Birkhoff-von-Neumann decomposition
    try:
        decomp = birkhoff.birkhoff_von_neumann_decomposition(probabilistic_ranking)
    except:
        decomp = birkhoff.approx_birkhoff_von_neumann_decomposition(probabilistic_ranking)

        if debug:
            print("Could get a approx decomposition with {}% accuracy".format(100*sum([x[0] for x in decomp])) )
            #print(probabilistic_ranking)

    return decomp
    p_birkhoff = np.asarray([np.max([0,x[0]]) for x in decomp])
    p_birkhoff /= np.sum(p_birkhoff)
    sampled_r = np.random.choice(range(len(decomp)), 1, p=p_birkhoff)[0]
    return np.argmax(decomp[sampled_r][1],axis=0)
    #except:
    #    return random_sampler(probabilistic_ranking)


def ideal_rank(users, item_affs):
    aff_prob = np.zeros(len(item_affs))
    for user in users:
        aff_prob += affinity_score_adv(user, item_affs)
        #aff_prob += scipy.stats.norm.pdf(item_affs, user,0.3)
        #aff_prob /= np.max(aff_prob)
        #aff_prob = np.around(aff_prob, 3)

    return aff_prob, (np.argsort(aff_prob)[::-1])


__main__()

