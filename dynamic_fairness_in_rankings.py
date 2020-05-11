##Imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import truncnorm

#from plotting import plot_contour, plot_neural_error, plot_click_bar_plot, group_item_clicks, plot_average_rank, \
#    plot_unfairness_over_time, plot_fairness_over_time, plot_ndcg, plot_NDCG_Unfairness, plot_with_errorbar

#from ranking_functions import pop_rank, IPS_rank, boost_rank, random_rank, controller_rank, probabilistic_rank, \
#    neural_rank, fair_rank, ideal_rank

"""pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],                    # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
}"""

# set global settings

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
mpl.rcParams.update(params)


def init_plotting():
    plt.rcParams['figure.figsize'] = (15,5)
    plt.rcParams['font.size'] = 25
    plt.rcParams['font.family'] = 'serif' #'Times New Roman'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    #plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['figure.subplot.wspace'] = 0.25
    plt.rcParams['figure.subplot.right'] = 0.95
    plt.rcParams['legend.columnspacing'] = 1
    #plt.rcParams['legend.handleheight'] = 4*plt.rcParams['legend.handleheight']
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

init_plotting()


mpl.use('Agg')
import numpy as np
import scipy.integrate
import scipy.stats
import random
#from birkhoff import birkhoff
#from sinkhorn_knopp import sinkhorn_knopp as skp
import pandas as pd
from IPython.display import clear_output
import time
import warnings; warnings.simplefilter('ignore') ##Ignores Warnings for nicer Plots. Disable for Debugging
# %matplotlib inline
import data_utils
import os
#from birkhoff import birkhoff_von_neumann_decomposition
import birkhoff
import relevance_network
from itertools import permutations


birkhoff.TOLERANCE = 10**(-8)



"""##Hyperparameter"""
DATA_SET = 2 #0 Synthetic old, 1 Jokes, 2 Movies
PLOT_PREFIX = "plots/"
U_ALPHA = 0.5
U_BETA = 0.5
U_STD = 0.3
W_FAIR = 10
KP = 0.01 #TODO maybe increase (increased from 0.001)
PROB_W = 5
LP_COMPENSATE_W = 1# 0.025 #1
#GROUP_BOUNDARIES = [[-1,-0.33],[0.33,1]] #Boundaries for Left and Right
GROUP_BOUNDARIES = [[-1,-0],[0,1]] #Boundaries for Left and Right
#MOVIE_RATING_FILE = "data/movie_data_binary_latent_5Comp_trial0.npy"#"data/movie_data_prepared_20features.npy"
MOVIE_RATING_FILE = "data/movie_data_binary_latent_5Comp_MostRatings_trial0.npy"#"data/movie_data_prepared_20features.npy"
HIDDEN_UNITS = 64
eps_m = 0
eps_p = 1

"""## Item Class"""


class Item:

    def __init__(self, polarity, quality=1, news_group = None, id=0):
        self.p = polarity
        self.q = quality
        self.id = id
        if (GROUP_BOUNDARIES[0][0] <= polarity <= GROUP_BOUNDARIES[0][1]):
            self.g = 0
        elif (GROUP_BOUNDARIES[1][0] <= polarity <= GROUP_BOUNDARIES[1][1]):
            self.g = 1
        else:
            self.g = 2
        self.news_group = news_group
    def get_features(self):
        tmp = [0] * 3
        tmp[self.g] = 1
        # return np.asarray([self.p,self.q, self.p**2] + tmp)
        return np.asarray([self.p, self.q] + tmp)


class Joke:
    def __init__(self, id, group=None):
        self.id = id
        if group is None:
            self.g = id
        else:
            self.g = group
    def get_features(self):
        return np.asarray([self.id])

class Movie:
    def __init__(self, id, group):
        self.id = id
        self.g = group

DATA_FAIR_FULLQUALITY = [Item(x,1) for x in np.linspace(-1,1,27)]
DATA_FAIR_BETTEREXTREME = [Item(x,max(abs(x),0.1)) for x in np.linspace(-1,1,27)]
DATA_FAIR_BETTERLEFT = [Item(x,max(0.1,abs((x-0.2)/1.2))) for x in np.linspace(-1,1,27)]
DATA_FAIR_BETTERLEFT2 = [Item(x,1) if x < -0.33 else Item(x,0.95) for x in np.linspace(-1,1,27)]

DATA_MORELEFT_FULLQUALITY = [Item(x,1) for x in np.linspace(-1,-0.4,15)] +[Item(x,1) for x in np.linspace(-0.3,0.3,5)] +[Item(x,1) for x in np.linspace(0.4,1,5)]
DATA_MORELEFT_BETTERRIGHT = [Item(x,max(0.1,abs((x+0.2)/1.2))) for x in np.linspace(-1,-0.4,15)] +[Item(x,max(0.1,abs((x+0.2)/1.2))) for x in np.linspace(-0.3,0.3,5)] +[Item(x,max(0.1,abs((x+0.2)/1.2))) for x in np.linspace(0.4,1,5)]
DATA_FAIR_DIVERSE = [Item(x,y) for x in np.linspace(-1,1,9) for y in np.linspace(0.11,0.99,3)]

DATASETS = {"FAIR_FULLQUALITY": DATA_FAIR_FULLQUALITY,"FAIR_BETTEREXTREME": DATA_FAIR_BETTEREXTREME,"FAIR_BETTERLEFT":DATA_FAIR_BETTERLEFT, "MORELEFT_FULLQUALITY":DATA_MORELEFT_FULLQUALITY,"MORELEFT_BETTERRIGHT": DATA_MORELEFT_BETTERRIGHT, "FAIR_DIVERSE":DATA_FAIR_DIVERSE}


"""##User Affinity and Distribution"""

def  binominal_dist(m1 = 0.5, std1 = 0.5, m2 = -0.5, std2 = 0.5, BI_LEFT = 0.5):
    if np.random.rand() < BI_LEFT:
        user = truncnorm.rvs(-1, 1, m1, std1, 1)
    else:
        user = truncnorm.rvs(-1, 1, m2, std2, 1)
    # std = u_std
    std = np.random.rand() / 2 + 0.05
    return np.asarray([user,std])
#Funktions for User score, position score, assigning groups and  User distributions
def affinity_score(user, items):
    return(affinity_score_adv(user,items, bernulli=False))

def affinity_score_adv(user, items, bernulli = True):
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

def assign_groups(items):
    n_groups = max([i.g for i in items])+1
    G = [ [] for i in range(n_groups)]
    for i, item in enumerate(items):
        G[item.g].append(i)
    return G

def sample_user_base(distribution = "bimodal", alpha =U_ALPHA, beta = U_BETA, u_std = U_STD, BI_LEFT = 0.5):
    if(distribution == "beta"):
        user = np.random.beta(alpha, beta)
        user *= 2
        user -= 1
        std = u_std
        #std = np.random.rand()*0.8 + 0.2
    elif(distribution == "discrete"):
        user = np.random.choice([-1,0,1])
        if(user == 0):
            std = 0.85
        else:
            std = 0.1
    elif(distribution == "bimodal"):
        #TODO changed standard deviation to 0.2
        if np.random.rand() < BI_LEFT:
            #user = truncnorm.rvs(-1,1,0.5,0.2,1)
            user = np.clip(np.random.normal(0.5,0.2,1),-1,1)
        else:
            #user = truncnorm.rvs(-1,1,-0.5,0.2,1)
            user = np.clip(np.random.normal(-0.5, 0.2, 1), -1, 1)
        #std = u_std
        std = np.random.rand()/2 + 0.05
    else:
        print("please specify a distribution for the user")
        return (0,1)
    return np.asarray([user, std]) #, user**2, np.sign(user))

def sample_user_joke():
    df, features = data_utils.load_data()

    while True:

        for i in range(df.shape[0]):
            yield (df.iloc[i].as_matrix(), features.iloc[i].as_matrix())
        print("All user preferences already given, restarting with the old user!")
        new_ordering = np.random.permutation(df.shape[0])
        df = df.iloc[new_ordering]
        features = features.iloc[new_ordering]

def sample_user_movie():
    ranking, features, _ = data_utils.load_movie_data_saved(MOVIE_RATING_FILE)
    print(np.shape(ranking))
    while True:
        random_order = np.random.permutation(np.shape(ranking)[0])
        #print(np.shape(random_order))
        for i in random_order:
            yield (ranking[i,:], features[i,:])
        print("All user preferences already given, restarting with the old user!")



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

def get_numerical_relevances_base_old(items, alpha = U_ALPHA, beta = U_BETA, std = U_STD):
    relevances = []
    #TODO numerical is not beta? STD different
    beta_dist = lambda x: scipy.stats.beta.pdf(x, alpha, beta)
    for item in items:
        aff = lambda x: affinity_score_adv((x,std), item )
        #aff = lambda x: np.exp(-(item - x)**2 / (2*u_std**2))
        rel = lambda x: aff(x) * 0.5* beta_dist((x+1)/2)
        integrated = scipy.integrate.quad(rel,-1,1)
        #print("Integration result", integrated)
        relevances.append(integrated[0])

    return np.asarray(relevances)

def get_numerical_relevances_base(items, alpha = U_ALPHA, beta = U_BETA, std = U_STD):
    users = [sample_user() for i in range(50000)]
    aff = [affinity_score(u, items) for u in users]

    return np.mean(np.asarray(aff), axis=0)

def get_numerical_relevances_joke(items):
    df, _ = data_utils.load_data()
    relevances = df.mean().as_matrix()
    #print("Numerical joke relevance: ", relevances)
    return np.asarray(relevances)

def get_numerical_relevances_movie(items):
    ranking , _, _ = data_utils.load_movie_data_saved(MOVIE_RATING_FILE)
    return  np.mean(ranking, axis=0) #Mean over all users


if DATA_SET == 1:
    sample_user_generator = sample_user_joke()
    sample_user = lambda: next(sample_user_generator)
    get_numerical_relevances = lambda x: get_numerical_relevances_joke(x)

elif DATA_SET == 0:
    sample_user = lambda : sample_user_base(distribution="bimodal")
    get_numerical_relevances = lambda x: get_numerical_relevances_base(x)

elif DATA_SET == 2:
    sample_user_generator = sample_user_movie()
    sample_user = lambda: next(sample_user_generator)
    get_numerical_relevances = lambda x: get_numerical_relevances_movie(x)


def random_sampler(probabilistic_ranking):
    n = np.shape(probabilistic_ranking)[0]
    additive = np.zeros((n,n))
    for i in range(n):
        additive[i,:] = np.sum(probabilistic_ranking[:i+1,:], axis=0)
    additive /= additive[-1,:][np.newaxis,:]
    for i in range(1000000):
        #Get the random item for each positions
        ranks = np.argmax(additive>np.random.rand(n)[np.newaxis,:], axis=0)
        if( len(ranks) == len(np.unique(ranks))):
            print("Took {} Samples to obtain a valid ranking".format(i))
            
            return ranks
    print("Could not sample in 10000 Iterations")


"""##Simulations"""

#CLICK FUNCTION P(click|item) = P(obs|position)*P(relevance|affiliation)*P(click|relevance) = position_bias*affiliation_bias
#rank by expected rel over users P(rel)
def click(user, popularity, items, weighted_popularity=None, G=None, ranking_method="Naive", click_model="PBM_log",cum_exposure = None, decomp = None, new_fair_rank = False, nn=None,  integral_fairness = None):
    n = len(popularity)
    click_prob = np.zeros(n)
    fairness_error = None
    
    #Ranking of the entries
    if(ranking_method=="Naive"):
        ranking = pop_rank(popularity)
        
    elif(ranking_method=="IPS"):
        assert(weighted_popularity is not None)
        ranking = IPS_rank(weighted_popularity)

    elif("IPS-LP" in ranking_method):
        #Try Linear Programm for fair ranking, when this fails, use last ranking
        if new_fair_rank or decomp is None:
            if(ranking_method=="Fair-E-IPS-LP"):
                group_fairness = get_unfairness(cum_exposure, weighted_popularity, G, error=False)
                decomp = fair_rank(items, weighted_popularity, debug=False, w_fair=W_FAIR, group_click_rel=group_fairness, impact=False)
            elif(ranking_method=="Fair-I-IPS-LP"):
                group_fairness = get_unfairness(popularity, weighted_popularity, G, error = False)
                decomp = fair_rank(items, weighted_popularity, debug=False, w_fair=W_FAIR, group_click_rel = group_fairness, impact=True)
            else:
                raise Exception("Unknown Fair method specified")
            
        if decomp is not None:
            p_birkhoff = np.asarray([np.max([0,x[0]]) for x in decomp])
            p_birkhoff /= np.sum(p_birkhoff)
            sampled_r = np.random.choice(range(len(decomp)), 1, p=p_birkhoff)[0]
            ranking = np.argmax(decomp[sampled_r][1],axis=0)
        else:
            ranking = IPS_rank(weighted_popularity)
            
    elif(ranking_method=="Fair-I-IPS"):
        fairness_error = get_unfairness(popularity, weighted_popularity, G, error = True)
        ranking = controller_rank(weighted_popularity, fairness_error)

    elif (ranking_method == "Fair-E-IPS"):
        fairness_error = get_unfairness(cum_exposure, weighted_popularity, G, error=True)
        ranking = controller_rank(weighted_popularity, fairness_error)

    elif("Pers" in ranking_method):
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

    #create prob of click based on position
    pos = position_bias(n, click_model, weighted_popularity[ranking])

    #reorder position probabilities to match popularity order
    pos_prob = np.zeros(n)
    pos_prob[ranking] = pos
    

    return pos_prob, ranking, decomp, fairness_error

def get_unfairness(clicks, rel, G, error = False, boost = False, group_sum = False):
    n = len(clicks)
    group_clicks = [sum(clicks[G[i]]) for i in range(len(G))]
    group_rel = [max(0.0001,sum(rel[G[i]])) for i in range(len(G))]
    group_fairness = [group_clicks[i]/group_rel[i] for i in range(len(G))]
    
    if(error and not boost):
        best = np.max(group_fairness)
        fairness_error = np.zeros(n)
        for i in range(len(G)):
            fairness_error[G[i]] = best - group_fairness[i]
        return fairness_error
    elif(error and boost):
        best = np.max(group_fairness)
        fairness_error = np.zeros(n)
        for i in range(len(G)):
            fairness_error[G[i]] = best / group_fairness[i]
        return fairness_error
    else:
        return group_fairness

#simulation function returns number of iterations until convergence
def simulate(popularity, items, ranking_method="Naive", click_model="PBM_log", iterations = 2000, numerical_relevance = None, head_start = -1):
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
    weighted_popularity = np.asarray(popularity,dtype = np.float32)
    popularity = np.asarray(popularity)
    pophist = np.zeros((iterations,len(items)))
    w_pophist = np.zeros((iterations,len(items)))
    if "Pers" in ranking_method:
        p_pophist = np.zeros((iterations,len(items)))
    else:
        p_pophist = None
    users = []
    #aff_scores = []
    aff_scores = np.zeros((iterations,len(items)))
    relevances = np.zeros(len(items))
    cum_exposure = np.zeros(len(items))
    hist = np.zeros((iterations,len(popularity)))
    prev = np.argsort(popularity)
    decomp = None
    group_prop = np.zeros((iterations,len(G)))
    group_clicks = np.zeros((iterations,len(G)))
    group_rel = np.zeros((iterations,len(G)))
    true_group_rel = np.zeros((iterations,len(G)))
    cum_fairness_error = np.zeros(len(items))
    NDCG = np.zeros(iterations)
    if(numerical_relevance is None):
        numerical_relevance = get_numerical_relevances(items)
    #params
    threshold = 100
    
    #counters
    count = 0 
    nn_errors = np.zeros(iterations)
    
    nn = None

    #x_test = np.asarray(list(map(lambda x: x.get_features(), items)))
    for i in range(iterations):
        count+=1

        #choose user
        if(i <= head_start*2):
            if i == head_start*2:
                sample_user = lambda: sample_user_base(BI_LEFT=0.5)
            elif i < head_start:
                sample_user = lambda : sample_user_base("bimodal", BI_LEFT=0)
            else:
                sample_user = lambda : sample_user_base("bimodal", BI_LEFT=1)


        user = sample_user()
        users.append(user)
        aff_probs = affinity_score_adv(user, items)
        relevances += aff_probs

        #clicking probabilities
        propensities, ranking, decomp, fairness_error = click(user, popularity, items, weighted_popularity / count, G, ranking_method, click_model, cum_exposure, decomp, count%100 == 9, nn=nn, integral_fairness = cum_fairness_error/count)

        #update popularity
        popularity, weighted_popularity = simulate_click(aff_probs, propensities, popularity, weighted_popularity, ranking, click_model)

        #Save History
        aff_scores[i] = aff_probs
        hist[i, :] = ranking
        cum_exposure += propensities
        pophist[i, :] = popularity
        w_pophist[i, :] = weighted_popularity

        #update neural network
        if "Pers" in ranking_method:
            if(i == 99): # Initialize Neural Network
                if DATA_SET == 0:
                    train_x = np.asarray(users)
                elif DATA_SET == 2 or DATA_SET ==1:
                    train_x = np.asarray([u[1] for u in users])
                if not "Skyline" in ranking_method:
                    nn = relevance_network.relevance_estimating_network(np.shape(train_x)[1], output_dim=len(items),
                                                                        hidden_units=HIDDEN_UNITS, joke=DATA_SET==1, news=(DATA_SET==0 or DATA_SET==2), logdir=PLOT_PREFIX)
                    train_y = w_pophist[:i + 1] - np.concatenate((np.zeros((1, len(items))), w_pophist[:i]))
                else:
                #Supervised Baseline
                    nn = relevance_network.relevance_estimating_network(np.shape(train_x)[1], output_dim=len(items),
                                                                        hidden_units=HIDDEN_UNITS, joke=DATA_SET==1, news=(DATA_SET==0 or DATA_SET==2), supervised=True, logdir=PLOT_PREFIX)
                    train_y = aff_scores[:i+1]
                nn.train(train_x, train_y, epochs=2000, trial=i)
            elif(i >99 and i %10 == 9):
                if "Skyline" in ranking_method:
                    train_y = aff_scores[:i+1]
                else:
                    train_y = np.concatenate((train_y, w_pophist[i - 9:i + 1] - w_pophist[i - 10:i]))
                    #assert(np.array_equal(train_y,w_pophist[:i + 1] - np.concatenate((np.zeros((1, len(items))), w_pophist[:i]))))
                if DATA_SET == 1 or DATA_SET == 2:
                    train_x = np.concatenate((train_x, np.asarray([u[1] for u in users[-10:]]) ))
                else:
                    train_x = np.concatenate((train_x, np.asarray([u for u in users[-10:]])))

                nn.train(train_x, train_y, epochs=10, trial=i)
                #TODO, changed the epoch number from 100

            if DATA_SET and i >= 99:
                predicted_relevances = nn.predict(user[1])
            elif i>= 99:
                predicted_relevances = nn.predict(user)
            if i>=99:
                nn_errors[i] = np.mean((predicted_relevances - aff_probs) ** 2)
                p_pophist[i, :] = predicted_relevances
            else:
                p_pophist[i, :] = weighted_popularity

        #Save statistics
        if(fairness_error is not None):
            cum_fairness_error += fairness_error
            
        if DATA_SET:
            NDCG[i] = get_ndcg_score(ranking, user[0])
        else:
            #NDCG[i] = get_ndcg_score(ranking, relevances/count) #numerical_relevance)
            NDCG[i] = get_ndcg_score(ranking, aff_probs)  # numerical_relevance)
            #NDCG[i] = get_ndcg_score(ranking, numerical_relevance) # TODO now we use the numerical Relevance instead of personalized
        #NDCG[i] = get_ndcg_score(ranking, relevances/count)
        #print( np.sum(np.abs(numerical_relevance - (relevances/count))))
        
        group_prop[i,:] = [np.sum(cum_exposure[G[i]]) for i in range(len(G))]
        group_clicks[i,:] =  [np.sum(popularity[G[i]]) for i in range(len(G))]
        #group_rel[i,:] = [sum(weighted_popularity[G[i]])/count for i in range(len(G))] #Normalizing by Count to have time normalized relevances
        #true_group_rel[i,:] = [sum(relevances[G[i]])/count for i in range(len(G))]
        if ("Pers" in ranking_method):
            group_rel[i,:] = [np.sum(p_pophist[i, G[g]]) for g in range(len(G))]
        elif("Naive" in ranking_method):
            group_rel[i, :] = [np.sum(pophist[i, G[g]]) for g in range(len(G))]
        else:
            group_rel[i,:] = [np.sum(weighted_popularity[G[g]]) for g in range(len(G))] # Having the sum of weighted clicks. Nicer Plots for the fairness measure
        #true_group_rel[i,:] = [sum(relevances[G[i]]) for i in range(len(G))]
        #TODO true rel News
        true_group_rel[i, :] = [np.sum(numerical_relevance[G[g]])*count for g in range(len(G))] # Try to avoid the spikes in the beginning
        
        
        prev = np.copy(ranking)



    #rank by expected rel over users P(rel)
    #calculate "ideal" rank
    #users = list of "affiliation scores"
    ideal_vals, ideal_ranking = ideal_rank(users, items)
    
    mean_relevances = relevances / count
    mean_exposure = cum_exposure / count
    
    #true_group_rel[:10,:]=true_group_rel[10,:][np.newaxis,:] 
    fairness_hist = {"prop": group_prop, "clicks": group_clicks, "rel": group_rel, "true_rel": true_group_rel, "NDCG": NDCG}
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
        #noise_rand_var = np.random.rand(len(aff_probs))
        viewed = rand_prop < propensities
        clicks = np.logical_and(rand_var < aff_probs, viewed)
        #clicks = np.logical_and(np.logical_and(rand_var < aff_probs, viewed), noise_rand_var <= eps_p)
        #un_rel_click = np.logical_and(noise_rand_var < eps_m, viewed)
        #clicks = np.logical_or(clicks, un_rel_click)
        popularity += clicks
        weighted_popularity += clicks / propensities
        #weighted_popularity += (clicks-eps_m)/(eps_p-eps_m) / propensities

    elif click_model=="Cascade" or click_model=="DCM":
      c_stop = 1  
      if click_model == "Cascade":
          gamma_click = 0
          gamma_no = 1
      else:
          gamma_click = 0.5
          gamma_no = 0.98
      for i,r in enumerate(ranking):
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
        assert(1==2)
    return popularity, weighted_popularity

"""#Testing

##Testing function
"""
def test_ranking(items, start_popularity, trials = 100, methods = ["Naive","IPS"], click_models = ["PBM_log"], save=False, iterations = 2000):
    G = assign_groups(items)
    if( len(G) <=3):

        group_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
    else:
        group_dict = dict([(i,str(x) ) for i, x in enumerate(G)])
    frac_c = [[],[],[]]
    rel_diff = []
    statistics = {"rel": [], "clicks": [], "exposure":[]}
    NDCG = []
    result = pd.DataFrame(columns=["Model","Group","Size","Top3","Rel","TrueRel","Exposure","Clicks","DiffRel","Exp/Rel","Click/Rel", "NDCG"])
    numerical_relevance = get_numerical_relevances(items)
    for click_model in click_models:
        for method in methods:
            iters = []
            top = np.zeros(len(G))
            rel = np.zeros(len(items))
            clicks = np.zeros(len(items))
            estimated_relevance = np.zeros(len(items))
            exposure = np.zeros(len(items))
            avg_linErrors = 0
            trial_NDCG = np.zeros(iterations)
            
            #for i in tqdm(range(trials)):
            for i in range(trials):
                popularity = np.copy(start_popularity)
                iterations, ranking_hist, popularity_hist, final_ranking, users, ideal, mean_relevances, w_pophist, errors, mean_exposure, fairness_hist, p_pophist = \
                simulate(popularity, items, ranking_method=method, click_model=click_model, iterations=iterations, numerical_relevance = numerical_relevance)
                iters.append(iterations)
                for r in final_ranking[:3]:
                    for g in range(len(G)):
                        top[g] += 1 if r in G[g] else 0
                        
                rel += mean_relevances
                clicks += popularity_hist[-1]
                exposure += mean_exposure
                trial_NDCG += fairness_hist["NDCG"]
                if("Naive" in method):
                    estimated_relevance += popularity_hist[-1]/iterations
                else:
                    estimated_relevance += w_pophist[-1]/iterations
                avg_linErrors += errors/iterations
            clicks /= (np.sum(iters))
            rel /= trials
            estimated_relevance /= trials
            exposure /= trials
            NDCG.append(trial_NDCG / trials)
            """
            print("  \n \n  Using ", method, " ranking, and ", click_model, " clickmodel we have the following statistics:")
            print("Avg num Iterations:{0:.2f} ;  in top 3; Pos:{1:.2f} Neg:{2:.2f} Neutral:{3:.2f}".format(np.mean(iters), top[1]/trials, top[0]/trials, top[2]/trials))
            #print("Average True relevances:" , rel )
            #print("Estimated relevance:", np.round(estimated_relevance,4))
            print("Difference between True relevance and estimation: {0:.5f}".format(np.sum(np.abs(rel-  estimated_relevance))))
            print("Total clicks: {0:.2f} ; Neg: {1:.2f} , Pos: {2:.2f} , Neutral {3:.2f}".format(sum(clicks),sum(clicks[G[0]]),sum(clicks[G[1]]),sum(clicks[G[2]])))
            #print("And {}% of the Linear Programm ranking failed".format(avg_linErrors/trials * 100))
            print("Exposure/Relevances per group : Neg: {0:.4f}, Pos: {1:.4f}, Neutral: {2:.4f}".format(*[sum(exposure[G[i]]) / sum(rel[G[i]]) for i in range(len(G))]))
            print("Clicks/Relevances per group : Neg: {0:.4f}, Pos: {1:.4f}, Neutral: {2:.4f}".format(*[sum(clicks[G[i]]) / sum(rel[G[i]]) for i in range(len(G))]))
            sys.stdout.flush()
            """
            for i, group in group_dict.items():
                result.loc[len(result.index)] = [method+ "&" + click_model.replace("_log",""), group, len(G[i]),
                                  top[i]/trials,sum(estimated_relevance[G[i]]), sum(rel[G[i]]), sum(exposure[G[i]]),
                                  sum(clicks[G[i]]),np.sum(np.abs(rel[G[i]]-estimated_relevance[G[i]])),
                                  sum(exposure[G[i]]) / sum(rel[G[i]]), sum(clicks[G[i]]) / sum(rel[G[i]]),NDCG[-1][-1]]
                 
            statistics["rel"].append(rel)
            statistics["clicks"].append(clicks)
            statistics["exposure"].append(exposure)
            for i in range(3):
                frac_c[i].append(sum(clicks[G[i]]))
    clear_output(wait=True)
    
    
    #labels = [ a + "\n" + b for a in click_models for b in methods]
    labels = [b for a in click_models for b in methods]
    plot_click_bar_plot(frac_c, labels, save)
    
    for i in range(len(NDCG)):
        plot_ndcg(NDCG[i], labels= labels[i], plot=False, window_size=30)
    plt.legend()
    plt.show()
    return result
    #grouped_bar_plot(statistics, G, [ a + "\n" + b for a in click_models for b in methods])

#Function that simulates and monitor the convergence to the relevance + the developement of cummulative fairness
def collect_relevance_convergence(items, start_popularity, trials = 10, methods = ["Naive","IPS"], click_models = ["PBM_log"], iterations = 2000, plot_individual_fairness = True, multiple_items=None):
    global KP
    global W_FAIR
    global MOVIE_RATING_FILE
    global sample_user, get_numerical_relevances

    rel_diff = []
    if multiple_items is None:
        G = assign_groups(items)
    else:
        if multiple_items == -1:
            G = assign_groups(items)
        else:
            assert(len(multiple_items) == trials)
            G = assign_groups(multiple_items[0])
    overall_fairness = np.zeros((len(click_models)*len(methods), trials, iterations, 4))
    pair_group_combinations = [(a,b) for a in range(len(G)) for b in range(a+1,len(G))]
    count=0
    run_data = []
    frac_c = [[] for i in range(len(G))]
    nn_errors = []
    method_dict = {"Naive": "Naive", "IPS": r'$\hat{R}^{IPS}(d)$', "Pers": "D-ULTR", "Skyline-Pers": "Skyline", "Fair-I-IPS": "FairCo(Imp)"}
    for click_model in click_models:

        #TODO very Hacky
        if "lambda" in click_model:
            lam = float(click_model.replace("lambda",""))
            KP = lam
            W_FAIR = lam
            click_model = "PBM_log"
        for method in methods:
            start_time = time.time()
            rel_diff_trial = []
            #rel_diff_top20 = []
            fairness = {"prop": np.zeros((trials,iterations,len(G))), "clicks": np.zeros((trials,iterations,len(G))), "rel": np.zeros((trials,iterations,len(G))), "true_rel": np.zeros((trials,iterations,len(G))), "NDCG": np.zeros((trials,iterations))}
            nn_error_trial=[]
            for i in range(trials):
                if multiple_items is not None:
                    if multiple_items==-1: #Load a new bernully relevance table
                        #MOVIE_RATING_FILE = "data/movie_data_binary_latent_HighestVar_trial{}.npy".format(i)
                        MOVIE_RATING_FILE = "data/movie_data_binary_latent_5Comp_trial{}.npy".format(i)
                        sample_user_generator = sample_user_movie()
                        sample_user = lambda: next(sample_user_generator)
                        get_numerical_relevances = lambda x: get_numerical_relevances_movie(x)

                    else:
                        items = multiple_items[i]
                        G = assign_groups(items)
                popularity = np.copy(start_popularity)
                #Run Simulation
                iterations, ranking_hist, popularity_hist, final_ranking, users, ideal, mean_relevances, w_pophist, errors, mean_exposure, fairness_hist, p_pophist = \
                simulate(popularity, items, ranking_method=method, click_model=click_model, iterations = iterations)
                ranking_hist = ranking_hist.astype(int)
                if "Pers" in method:
                    nn_error_trial.append(errors)

                #Calculate the relevance difference between true relevance and approximation
                # Diff = |rel' - rel|
                if method=="Naive":
                    rel_estimate = popularity_hist / np.arange(1, iterations+1)[:,np.newaxis]
                elif "Pers" in method  :
                    p_pophist[99:,:] = [np.sum(p_pophist[98:100+i,:], axis=0) for i in range(len(p_pophist)-99)]

                    rel_estimate = p_pophist / (np.arange(iterations)+1)[:,np.newaxis]
                else:
                    rel_estimate = w_pophist / np.arange(1, iterations+1)[:, np.newaxis]

                rel_diff_trial.append( np.mean(np.abs(rel_estimate - (mean_relevances)[np.newaxis,:]), axis=1))
                #if len(items) > 20 and False:
                #        rel_top20 = np.mean(np.abs(rel_estimate[:,ranking_hist[:,:20]] - mean_relevances[np.newaxis,ranking_hist[:,:20]]),axis = 1)
                #        rel_diff_top20.append(rel_top20)



                #Cummulative Fairness per Iteration summed over trials     
                for key, value in fairness_hist.items():
                    fairness[key][i] = value
                if(trials <=1):
                    #Plot Group Clicks and Items Average Rank
                    group_item_clicks(popularity_hist[-1], G)
                    plot_average_rank(ranking_hist, G)
                    print("Relevance Difference: ", np.sum((mean_relevances - rel_estimate[-1]) ** 2))

                    #Plot Ranking History
                    plt.title("Ranking History")
                    plt.axis([0, iterations, 0,len(items)])
                    if len(G) <=3:
                        group_colors = {0:"blue",1:"red",2:"black"}
                        group_labels = {0:"Negative",1:"Positive",2:"black"}
                    else:
                        group_colors = [None for i in range(len(G))]
                    item_rank_path = np.ones((iterations,len(items)))
                    for i in range(iterations):
                        item_rank_path[i,ranking_hist[i,:]]=np.arange(len(items))
                    for i in range(len(items)):
                        group_color_i = group_colors[[x for x in range(len(G)) if i in G[x]][0]]
                        plt.plot(np.arange(iterations),item_rank_path[:,i], color=group_color_i)

                    custom_lines = [Line2D([0], [0], color="blue", lw=4),
                                    Line2D([0], [0], color="red", lw=4)]

                    plt.legend(custom_lines, ['Negative', 'Positive'])
                    #plt.show()
                    plt.legend()
                    plt.savefig( PLOT_PREFIX + "Rankinghistory_"+click_model+"_"+method+".pdf", bbox_inches="tight")

            print("Time for " + click_model + " " + method+ " was: {0:.4f}".format(time.time()-start_time))

            if "Pers" in method:
                mean_trial_error = np.mean(np.asarray(nn_error_trial), axis=0)
                nn_errors.append(mean_trial_error)

            count += 1
            #Plot the Fairness per Group for a single model
            if(plot_individual_fairness):
                plot_fairness_over_time(fairness, G, method)
            #Collect Data for later
            run_data.append(fairness)

            for i in range(len(G)):
                frac_c[i].append(np.mean(fairness["clicks"][:,-1,i]) / iterations)
                
            if(len(rel_diff_trial)==1):
                rel_tmp = np.asarray(rel_diff_trial[0])
                rel_std = np.zeros(np.shape(rel_tmp))
            else:
                rel_tmp = np.mean(np.asarray(rel_diff_trial), axis=0)
                rel_std = np.std(np.asarray(rel_diff_trial), axis=0)
            rel_diff.append([rel_tmp,  method_dict[method], rel_std])
            #if(len(items)>20 and False):
            #    rel_diff.append((np.mean(np.asarray(rel_diff_top20),axis=0),click_model.replace("_log","") + " "+ method + "top 20"))


    np.save(PLOT_PREFIX+ "Fairness_Data.npy", run_data)
    #Plot NDCG
    plt.figure("NDCG")
    #plt.title("Average NDCG")
    #labels = [ a + "\n" + b for a in click_models for b in methods]
    labels = [ b for a in click_models for b in methods]
    for i, nd in enumerate(run_data):
        plot_ndcg(np.mean(nd["NDCG"], axis=0), label=labels[i], plot=False, window_size=100, std =nd["NDCG"])
    plt.legend()
    ax = plt.gca()
    plt.savefig(PLOT_PREFIX + "NDCG.pdf", bbox_inches="tight", dpi=800)
    plt.show()
    plt.close("all")
    
    #Plot Clicks
    plot_click_bar_plot(frac_c, labels, save=True)
    
    if True:
        plt.close("all")
        #Plot Convergence of Relevance
        for y in rel_diff:
            p = plt.plot(np.arange(len(y[0])), y[0], label=y[1])
            color = p[-1].get_color()
            plt.fill_between(np.arange(len(y[0])), y[0]- y[2],
                             y[0] + y[2], alpha=0.4, color=color)

        plt.legend(loc="best")
        plt.axis([0,len(y[0]),0,0.3])
        #plt.ylabel("Avg diff between \n True & Estimated Relevance  ")
        plt.ylabel(r'average $|\hat{R}(d) - {R}(d)|$')

        plt.xlabel("Users")
        plt.savefig(PLOT_PREFIX + "Relevance_convergence.pdf", bbox_inches="tight")
        plt.show()

    plot_neural_error(nn_errors, [b for a in click_models for b in methods if "Pers" in b])
    #Plot Unfairness over time between different models


    for i, data in enumerate(run_data):

        for a, b in pair_group_combinations:
            overall_fairness[i,:, :, 0] += np.abs(
                data["prop"][:,:, a] / data["rel"][:,:, a] - data["prop"][:,:, b] / data["rel"][:,:,b])
            overall_fairness[i,:, :, 1] += np.abs(
                data["prop"][:,:, a] / data["true_rel"][:,:, a] - data["prop"][:,:, b] / data["true_rel"][:,:, b])
            overall_fairness[i,:, :, 2] += np.abs(
                data["clicks"][:,:, a] / data["rel"][:,:, a] - data["clicks"][:,:, b] / data["rel"][:,:, b])
            overall_fairness[i,:, :, 3] += np.abs(
                data["clicks"][:,:, a] / data["true_rel"][:,:, a] - data["clicks"][:,:, b] /data["true_rel"][:,:, b])

    overall_fairness /= len(pair_group_combinations)
    plot_unfairness_over_time(overall_fairness, click_models, methods, True)


    fig, ax = plt.subplots()
    ax2 = None
    for i, data in enumerate(run_data):
        ax2 = plot_NDCG_Unfairness(data["NDCG"], overall_fairness[i, :, :, 1], ax =ax, ax2=ax2, label=labels[i], unfairness_label="Exposure Unfairness")
    ax.legend()
    plt.savefig(PLOT_PREFIX + "NDCG_UnfairExposure.pdf", bbox_inches="tight", dpi=800)
    plt.close("all")


    fig, ax = plt.subplots()
    ax2 = None
    for i, data in enumerate(run_data):
        ax2 = plot_NDCG_Unfairness(data["NDCG"], overall_fairness[i, :, :, 3], ax =ax, ax2=ax2, label=labels[i], unfairness_label="Impact Unfairness")
    ax.legend()
    plt.savefig(PLOT_PREFIX + "NDCG_UnfairImpact.pdf", bbox_inches="tight", dpi=800)
    plt.close("all")


def experiment_different_starts(load=False, prefix = "", news_items = False):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    methods = ["Naive", "IPS", "Fair-I-IPS"]
    click_model = "PBM_log"
    #starts = [0, 1, 2, 3, 4, 5, 7, 10, 15,25,35]
    starts = [0, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400]
    iterations = 3000
    trials = 50
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

def load_news_items(n = 30, completly_random = False, n_left=None):
    data_full, data_medium, data_tiny = data_utils.load_news_data()
    items = []

    if completly_random:
        for index, row in data_full.sample(n).iterrows():
            items.append(Item(row["Bias"], quality=1, id=index, news_group=row["Group"]))
    elif n_left is not None:
        c_left = 0
        for index, row in data_full.sample(frac=1).iterrows():
            if((row["Bias"]<0 and c_left < n_left ) or (row["Bias"]>0 and len(items) < n - (n_left -c_left))):
                items.append(Item(row["Bias"], quality=1, id=index, news_group=row["Group"]))
            if( len(items)>= n):
                return items

    else:
        for index, row in data_tiny.iterrows():
            items.append(Item(row["Bias"],quality=1, id=index, news_group=row["Group"]))

    return items


def squarify(fig):
    w, h = fig.get_size_inches()
    if w > h:
        t = fig.subplotpars.top
        b = fig.subplotpars.bottom
        axs = h*(t-b)
        l = (1.-axs/w)/2
        fig.subplots_adjust(left=l, right=1-l)
    else:
        t = fig.subplotpars.right
        b = fig.subplotpars.left
        axs = w*(t-b)
        l = (1.-axs/h)/2
        fig.subplots_adjust(bottom=l, top=1-l)

def get_unfairness_from_rundata(run_data, pair_group_combinations, trials=10, iterations=7000):
    overall_fairness = np.zeros((len(run_data), trials, iterations, 2))
    for i, data in enumerate(run_data):
        print("Shapes: Prop", np.shape(data["prop"]), "rel", np.shape(data["true_rel"]), "clicks", np.shape(data["clicks"]))
        for a, b in pair_group_combinations:
            overall_fairness[i, :, :, 0] += np.abs(
                data["prop"][:, :iterations, a] / data["true_rel"][:, :iterations, a] - data["prop"][:, :iterations, b] / data[
                                                                                                            "true_rel"][
                                                                                                        :, :iterations, b])
            overall_fairness[i, :, :, 1] += np.abs(
                data["clicks"][:, :iterations, a] / data["true_rel"][:, :iterations, a] - data["clicks"][:, :iterations, b] / data[
                                                                                                                "true_rel"][
                                                                                                            :, :iterations,
                                                                                                            b])
    overall_fairness /= len(pair_group_combinations)
    return overall_fairness


def test_different_population(trials = 10, iterations = 2000, load = False, prefix = "", news_items=False):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    global sample_user
    global get_numerical_relevances
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


def run_and_save_final_stats(methods, items, trials=10, iterations=2000, get_users = False, multi_items = None):
    click_model = "PBM_log"
    final_stats = np.zeros((len(methods),trials,3))
    run_data=[]
    if get_users:
        user_distribution = np.zeros((len(methods),trials,iterations))
    for m, method in enumerate(methods):
        for i in range(trials):

            print("Progress {0:.2f}%".format((m * trials  + i) / (len(methods) * trials) * 100))
            if multi_items is None:
                random.shuffle(items)
                G = assign_groups(items)
            else:
                items = multi_items[i]
                G = assign_groups(items)
            pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]
            popularity = np.ones(len(items))
            iterations, ranking_hist, popularity_hist, final_ranking, users, ideal, mean_relevances, w_pophist, errors, mean_exposure, fairness_hist, p_pophist= \
                simulate(popularity, items, ranking_method=method, click_model=click_model, iterations=iterations)

            final_stats[m,i,0] = np.mean(fairness_hist["NDCG"])
            if get_users:
                user_distribution[m,i,:] = np.asarray([u[0] for u in users]).flatten()
            for a, b in pair_group_combinations:
                final_stats[m, i, 1] += np.abs(fairness_hist["prop"][-1,a] / fairness_hist["true_rel"][-1, a] -
                                               fairness_hist["prop"][-1, b] / fairness_hist["true_rel"][-1,b])

                final_stats[m, i, 2] += np.abs(fairness_hist["clicks"][-1,a] / fairness_hist["true_rel"][-1, a] -
                                               fairness_hist["clicks"][-1, b] / fairness_hist["true_rel"][-1,b])


    final_stats[:,:,1:] /= len(pair_group_combinations)
    if get_users:
        return final_stats , user_distribution
    return final_stats
def news_experiment():
    global PLOT_PREFIX
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
                                  #methods=["Naive", "IPS", "Pers", "Fair-I-IPS", "Fair-I-Pers", "Fair-E-IPS", "Fair-E-Pers"], iterations=iterations,
                                  methods=["Naive", "IPS", "Fair-I-IPS", "Fair-E-IPS"], iterations=iterations,
                                  plot_individual_fairness=True, multiple_items=multiple_items)

def movie_experiment(plot_prefix, methods, rating_file, trials=4, iterations = 5000, binary_rel=False):
    global PLOT_PREFIX, MOVIE_RATING_FILE
    global sample_user, get_numerical_relevances
    MOVIE_RATING_FILE = rating_file
    PLOT_PREFIX = plot_prefix

    sample_user_generator = sample_user_movie()
    sample_user = lambda: next(sample_user_generator)
    get_numerical_relevances = lambda x: get_numerical_relevances_movie(x)
    _, _, groups = data_utils.load_movie_data_saved(MOVIE_RATING_FILE)
    items = []
    for i,g in enumerate(groups):
        items.append(Movie(i,g))
    popularity = np.ones(len(items))
    G = assign_groups(items)

    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)

    if binary_rel:
        multi = -1
    else:
        multi = None
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],#["PBM_log"],
                                  methods=methods, iterations=iterations,
                                  plot_individual_fairness=True, multiple_items=multi)

def __main__():
    global PLOT_PREFIX
    global sample_user
    global DATA_SET
    #overview = test_ranking(items, popularity, trials, click_models=["PBM_log"], methods=["Naive", "IPS", "P-Controller"], save = True, iterations=5000)
    #print(tabulate(overview, headers='keys', tablefmt='psql'))

    #experiment_different_starts(True)
    #load_and_plot_all()
    #test_different_population(10, 20000, True)
    #test_different_groups(40, 2000,False) #Maybe redo with more trials.
    #load_and_plot_lambda_comparison() #Runs the plotting of the Controller/LP for different lambdas

    #news_experiment()
    #load_and_plot_news()
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
                    #"Movie_data/plots/MovieExperimentFull_binary/",
                     ["Naive", "IPS", "Pers", "Skyline-Pers", "Fair-I-IPS"],
                     #["IPS", "Pers"],
                        #"data/movie_data_binary_latent_5Comp_trial0.npy",
                     "data/movie_data_binary_latent_5Comp_MostRatings_trial0.npy",
                        #"data/movie_data_prepared_20features.npy",
                     trials=1, iterations=6000, binary_rel = True)
    """
    movie_experiment("Movie_data/plots/BinaryMovies_Debug_5Comp/",
                     ["Pers",  "IPS"], # "Skyline-Pers"
                     # ["IPS", "Pers"],
                     "data/movie_data_binary_latent_5Comp_trial0.npy",
                     # "data/movie_data_prepared_20features.npy",
                     trials=1, iterations=6000, binary_rel=True)
    """
    #load_and_plot_movie_experiment()
    #DATA_SET = 0
    #test_different_groups(20, 3000, True, prefix="News_data/different_groups_binary_final2/")
    #experiment_different_starts(True, "News_data/different_starts_binary_final2/", news_items=True)
    #test_different_population(20, 3000, True, prefix="News_data/different_population_binary_final2/", news_items=True)

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

def plot_neural_error(errors, labels):
    plt.close("all")
    # Plot Neural error
    print(np.shape(errors), "Neural Network Error Shape")
    for i, error in enumerate(errors):
        plt.plot(np.arange(100, len(error) + 100), error, label=labels[i])
    plt.legend()
    plt.ylabel("Difference True and estimated relevance")
    plt.xlabel("Users")
    plt.savefig(PLOT_PREFIX + "Neural_Error.pdf", bbox_inches="tight")


def plot_click_bar_plot(frac_c, labels, save=False):
    color_loop = ["blue","red","black","green","orange"]
    group_colors = [color_loop[i%len(color_loop)] for i in range(len(frac_c))]
    if len(frac_c) <= 3:
        plot_labels = ["Negative", "Positive","Neutral"]
    else:
        plot_labels = ["Group {}".format(i) for i in range(len(frac_c))]
    n = len(labels)
    for i in range(len(frac_c)):
        if i == 0:
            plt.bar(np.arange(n), frac_c[i], color=group_colors[i], edgecolor='white', width=1, label=plot_labels[i])
            bottom = frac_c[i]
        else:
            plt.bar(np.arange(n), frac_c[i], bottom=bottom, color=group_colors[i], edgecolor='white', width=1, label=plot_labels[i])
            bottom = np.add(frac_c[i] ,bottom)



    plt.xticks(np.arange(n), labels, fontweight='bold')
    total_clicks = np.round(np.sum(np.asarray(frac_c), axis=0),
                            3)  # np.round(np.add(np.add(frac_c[0],frac_c[1]),frac_c[2]),3)
    for i in range(n):
        plt.text(x=i, y=total_clicks[i], s=total_clicks[i], size=10)
    plt.ylabel("Average number of clicks")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if (save):
        plt.savefig(PLOT_PREFIX + "Barplot.pdf", bbox_inches="tight")
    else:
        plt.show()


def grouped_bar_plot(statistics, groups, models):
    # set width of bar
    barWidth = 0.25
    for j in range(len(models)):
        x_pos = np.arange(3.)
        colors = ["black", "grey", "orange"]
        i = 0
        for key, value in statistics.items():
            v = np.asarray(value[j])

            plt.bar(x_pos, [sum(v[groups[i]]) for i in range(len(groups))], color=colors[i], width=barWidth,
                    edgecolor='white', label=key)
            x_pos += barWidth
            i += 1
        # Add xticks on the middle of the group bars
        plt.xlabel('Group', fontweight='bold')
        plt.xticks([r + barWidth for r in range(3)], ["Left", "Right", "Neutral"])
        plt.title(models[j])
        # Create legend & Show graphic
        plt.legend()
        plt.show()
        # plt.savefig(PLOT_PREFIX + "Proportionalitybar"+str(j)+ ".pdf")
        # files.download("Proportionalitybar"+str(j)+ ".pdf")
        plt.close("all")


def group_item_clicks(clicks, G):
    if len(G) <= 3:
        group_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
        color_dict = {0: "blue", 1: "red", 2: "black"}

        for i, g in enumerate(G):
            plt.plot(np.arange(len(g)), np.sort(clicks[g])[::-1], color=color_dict[i], label=group_dict[i], marker='o')
    else:

        for i, g in enumerate(G):
            plt.plot(np.arange(len(g)), np.sort(clicks[g])[::-1], label=i, marker='o')
    plt.xlabel("Item")
    plt.ylabel("Clicks")
    plt.title("Clicks per Item and Group")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    plt.close("all")


def plot_average_rank(ranking_hist, G):
    if len(G) <= 3:
        group_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
        color_dict = {0: "blue", 1: "red", 2: "black"}
    else:
        group_dict = dict([(i, x) for i, x in enumerate(G)])
        colors = None
    iterations, n = np.shape(ranking_hist)
    summed_ranks = np.zeros(n)
    for i in range(iterations):
        summed_ranks[ranking_hist[i]] += np.arange(1, n + 1)
    avg_rank = summed_ranks / iterations
    group_per_item = np.ones(n) * -1
    for i in range(len(G)):
        group_per_item[G[i]] = i
    labels = [group_dict[x] for x in group_per_item]
    if len(G) <= 3:
        colors = [color_dict[x] for x in group_per_item]
    plt.bar(np.arange(n), avg_rank, color=colors)
    plt.xlabel("Items")
    plt.ylabel("Average Rank")
    plt.title("Average Rank per Item")
    plt.show()
    plt.close("all")


def plot_unfairness_over_time(overall_fairness, click_models, methods, only_two=True):
    n = np.shape(overall_fairness)[2]
    # plt.figure("Unfairness",figsize=(16,4))
    if (only_two):
        plt.figure("Unfairness", figsize=(10, 4))
    else:
        plt.figure("Unfairness", figsize=(17, 4))
    fairness_std = np.std(overall_fairness, axis=1)
    overall_fairness = np.mean(overall_fairness, axis=1)
    for i in range(len(click_models) * len(methods)):
        if not only_two:
            plt.subplot(141)
            # plt.title("Group-Difference \n Exposure per est. Relevance")
            # plt.axis([0,n,0,3])
            p = plt.plot(np.arange(n), overall_fairness[i, :, 0],
                         label=methods[i % len(methods)])  # +" " + click_models[i//len(methods)])
            color = p[-1].get_color()
            plt.fill_between(np.arange(n), overall_fairness[i, :, 0] - fairness_std[i, :, 0],
                             overall_fairness[i, :, 0] + fairness_std[i, :, 0], alpha=0.4, color=color)

            plt.xlabel("Users")
            #plt.ylabel("Group-Difference of \n Exposure per est. Relevance")
            plt.ylabel("Exposure Unfairness")
            ax = plt.gca()
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
        if only_two:
            plt.subplot(121)
        else:
            plt.subplot(142)
        # plt.axis([0,n,0,1])
        # plt.title("Group-Difference \n Exposure per True Relevance")
        p = plt.plot(np.arange(n), overall_fairness[i, :, 1],
                     label=methods[i % len(methods)])  # +" " + click_models[i//len(methods)])
        color = p[-1].get_color()
        plt.fill_between(np.arange(n), overall_fairness[i, :, 1] - fairness_std[i, :, 1],
                         overall_fairness[i, :, 1] + fairness_std[i, :, 1], alpha=0.4, color=color)
        ax = plt.gca()
        ax.set_ylim(0, np.max(overall_fairness[:, 15:, 1]))
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        plt.xlabel("Users")
        #plt.ylabel("Group-Difference of \n Exposure per True Relevance")
        plt.ylabel("Exposure Unfairness")
        if not only_two:
            plt.subplot(143)
            # plt.axis([0,n,0,1])
            # plt.title("Group-Difference \n Clicks per est. Relevance")
            p = plt.plot(np.arange(n), overall_fairness[i, :, 2],
                         label=methods[i % len(methods)])  # +" " + click_models[i//len(methods)])
            color = p[-1].get_color()
            plt.fill_between(np.arange(n), overall_fairness[i, :, 2] - fairness_std[i, :, 2],
                             overall_fairness[i, :, 2] + fairness_std[i, :, 2], alpha=0.4, color=color)

            plt.xlabel("Users")
            #plt.ylabel("Group-Difference of \n Clicks per est. Relevance")
            plt.ylabel("Impact Unfairness")
            ax = plt.gca()
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
        if only_two:
            plt.subplot(122)
        else:
            plt.subplot(144)
        # plt.title("Group-Difference \n Clicks per True Relevance")
        p = plt.plot(np.arange(n), overall_fairness[i, :, 3],
                     label=methods[i % len(methods)])  # +" " + click_models[i//len(methods)])
        color = p[-1].get_color()
        plt.fill_between(np.arange(n), overall_fairness[i, :, 3] - fairness_std[i, :, 3],
                         overall_fairness[i, :, 3] + fairness_std[i, :, 3], alpha=0.4, color=color)

        # plt.axis([0,n,0,1])
        ax = plt.gca()
        ax.set_ylim(0, np.max(overall_fairness[i, 15:, 3]))
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        plt.xlabel("Users")
        #plt.ylabel("Group-Difference of \n Clicks per True Relevance")
        plt.ylabel("Impact Unfairness")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(PLOT_PREFIX + "Unfairness.pdf", bbox_inches="tight", dpi=800)
    # files.download("Unfairness.pdf")

    plt.show("Unfairness")


def plot_fairness_over_time(fairness, G, model):
    n = np.shape(fairness["rel"])[1]
    if len(G) <=3:
        group_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
    else:
        group_dict = [str(g) for g in G]
    color_dict = {0: "blue", 1: "red", 2: "black", 3: "orange", 4: "yellow", 5: "purple", 6: "brown"}
    plt.figure("Fairness", figsize=(17, 4))

    print("The difference in Relevance:",  np.mean((fairness["rel"] - fairness["true_rel"])**2, axis=(0,1)))
    print("Mean True Relevance:", np.mean(fairness["true_rel"],axis=(0,1)))
    print("Mean est Relevance:", np.mean(fairness["rel"], axis=(0, 1)))
    for g in range(len(G)):
        plt.subplot(141)
        # plt.axis([0,n,0,3])
        plt.title("Model " + model + "\n Exposure per est. Relevance")
        plt.plot(np.arange(n), np.mean(fairness["prop"][:,:, g] / fairness["rel"][:,:, g], axis=0), label="Group " + group_dict[g],
                 color=color_dict[g])
        plt.subplot(142)
        # plt.axis([0,n,0,1])
        plt.title("Model " + model + "\n Exposure per True Relevance")
        plt.plot(np.arange(n), np.mean(fairness["prop"][:,:, g] / fairness["true_rel"][:,:, g], axis=0), label="Group " + group_dict[g],
                 color=color_dict[g])
        plt.subplot(143)
        # plt.axis([0,n,0,1])
        plt.title("Model " + model + "\n Clicks per est. Relevance")
        plt.plot(np.arange(n), np.mean(fairness["clicks"][:,:, g] / fairness["rel"][:,:, g], axis=0), label="Group " + group_dict[g],
                 color=color_dict[g])
        plt.subplot(144)
        plt.title("Model " + model + "\n Clicks per True Relevance")
        plt.plot(np.arange(n), np.mean(fairness["clicks"][:,:, g] / fairness["true_rel"][:,:, g], axis=0), label="Group " + group_dict[g],
                 color=color_dict[g])
        #plt.axis([0, n, 0, 1])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(PLOT_PREFIX + "Fairness" + model + ".pdf", bbox_inches="tight")
    plt.show()
    plt.close("all")
    # files.download("Fairness"+ model+ ".pdf")


def plot_ndcg(ndcg, label="", plot=True, figure_name="NDCG", window_size=0, std=None, color=None):
    plt.figure(figure_name)
    #TODO changed from cummulative to moving average
    if window_size > 0:

        #moving_average = np.asarray([np.mean(ndcg[max(0, i-window_size):i+1]) for i in range(len(ndcg))])
        #moving_average = np.asarray([np.mean(ndcg[i:i+window_size]) for i in range(len(ndcg)-window_size)])
        # moving_average = np.convolve(ndcg, np.ones((window_size))/window_size, mode='valid')
        # moving_average = [np.mean(ndcg[:(i+1)]) for i in range(len(ndcg))]
        moving_average = np.cumsum(ndcg) / np.arange(1, len(ndcg) + 1)
        p = plt.plot(np.arange(len(moving_average)), moving_average, label=label, linestyle='-', color=color)
        #plt.axis([0, len(moving_average), 0.75, 1])
        plt.axis([0, len(moving_average), 0.65, 1])
        print(np.shape(ndcg), np.shape(moving_average), np.sum(ndcg), np.sum(moving_average))

        plt.ylabel("Avg. Cumulative NDCG")
        if (std is not None):
            if color is None:
                color = p[-1].get_color()
            # print(np.shape(std))
            std = np.std([np.mean(std[:,max(0, i - window_size):i+1], axis=1) for i in range(len(ndcg))], axis=1)
            #std = np.std([np.mean(std[:,i:i + window_size], axis=1) for i in range(len(ndcg) - window_size)], axis=0)
            #std = np.std(np.cumsum(std, axis=1) / np.arange(1, len(ndcg) + 1), axis=0)
            # print(np.shape(std),np.shape(ndcg),np.shape(moving_average))
            plt.fill_between(np.arange(len(moving_average)), moving_average - std, moving_average + std, alpha=0.4,
                             color=color)
    else:
        p = plt.plot(np.arange(len(ndcg)), ndcg, label=label)
        # plt.axis([0, len(ndcg), 0.75, 0.9])
        plt.axis([0, len(ndcg), 0.9, 1])

        if (std is not None):
            color = p[-1].get_color()
            plt.fill_between(np.arange(len(ndcg)), ndcg - std, ndcg + std, alpha=0.4,
                             color=color)

        plt.ylabel("NDCG")
    plt.xlabel("Users")
    if (plot):
        plt.title("Average NDCG in model " + label)
        plt.show()


def plot_NDCG_Unfairness(ndcg, unfairness, ax, ax2=None, label="", unfairness_label="Unfairness", synthetic=False, color=None):
    # fig, ax =plt.subplots()
    # plt.figure(figure_name)
    trials, n = np.shape(ndcg)

    cum_ndcg = np.cumsum(ndcg, axis=1) / np.arange(1, n + 1)
    std_ndcg = np.std(cum_ndcg, axis=0)
    cum_ndcg = np.mean(cum_ndcg, axis=0)

    unfairness_std = np.std(unfairness, axis=0)
    unfairness = np.mean(unfairness, axis=0)
    p = ax.plot(np.arange(n), cum_ndcg, label=label, linestyle='-', color=color)
    if color is None:
        color = p[-1].get_color()
    ax.set_xlim([0, n])
    if (synthetic):
        ax.set_ylim([0.95, 1])
    else:
        ax.set_ylim([0.75, 0.9])
    if ax2 is not None:
        ax2.set_xlim([0, n])
        if (synthetic):
            ax2.set_ylim([0, 0.2])
        else:
            ax2.set_ylim([0, 0.2])
        ax2.set_xlabel("Users")
        ax2.set_ylabel(unfairness_label)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    ax.fill_between(np.arange(n), cum_ndcg - std_ndcg, cum_ndcg + std_ndcg, alpha=0.4, color=color)
    ax.set_xlabel("Users")
    ax.set_ylabel("Avg. Cumulative NDCG")

    if ax2 is not None:
        # ax2 = ax.twinx()
        x0, x1 = ax2.get_xlim()
        y0, y1 = ax2.get_ylim()
        # ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        # ax2.set_aspect("equal")
        ax2.plot(np.arange(n), unfairness, label=label, color=color)  # , linestyle=':',
        ax2.fill_between(np.arange(n), unfairness - unfairness_std, unfairness + unfairness_std, alpha=0.4, color=color)
    return ax2


def plot_numerical_relevance(items, alpha=U_ALPHA, beta=U_BETA, u_std=U_STD):
    relevances = []
    beta_dist = lambda x: scipy.stats.beta.pdf(x, alpha, beta)
    for item in items:
        aff = lambda x: affinity_score_adv((x, u_std), item)
        # aff = lambda x: np.exp(-(item - x)**2 / (2*u_std**2))
        rel = lambda x: aff(x) * 0.5 * beta_dist((x + 1) / 2)
        integrated = scipy.integrate.quad(rel, -1, 1)
        # print("Integration result", integrated)
        relevances.append(integrated)
    acc = 25
    plot_relevances = np.zeros(acc)
    for i, j in enumerate(np.linspace(-1, 1, acc)):
        aff = lambda x: affinity_score_adv((x, u_std), j)
        rel = lambda x: aff(x) * 0.5 * beta_dist((x + 1) / 2)
        plot_relevances[i] = scipy.integrate.quad(rel, -1, 1)[0]
    plt.plot(np.linspace(-1, 1, acc), plot_relevances)
    plt.axis([-1, 1, 0, 0.5])
    plt.ylabel("Relevance")
    plt.xlabel("Documents Affinity")
    plt.show()
    return np.asarray(relevances)

def load_and_plot_movie_experiment(trials = 10, n_user = 6000):
    global DATA_SET
    DATA_SET = 2

    _, _, groups = data_utils.load_movie_data_saved("data/movie_data_binary_latent_5Comp_trial0.npy")
    items = []
    for i,g in enumerate(groups):
        items.append(Movie(i,g))
    G = assign_groups(items)
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]
    #movie_data = np.load("Movie_data/plots/MovieExperimentFull/Fairness_Data.npy")
    #movie_data = np.load("Movie_data/plots/MovieExperimentFull_Var/Fairness_Data.npy", allow_pickle=True)
    #movie_data = np.load("Movie_data/plots/MovieExperimentFull_HighestVar/Fairness_Data.npy", allow_pickle=True)
    #movie_data = np.load("Movie_data/plots/MovieExperimentFull_binary/Fairness_Data.npy")
    movie_data = np.load("Movie_data/plots/MovieExperimentFull_5Comanies/Fairness_Data.npy", allow_pickle=True)
    #labels = np.asarray(["Naive", "IPS", "Pers", "Skyline-Pers", "Fair-I-IPS", "Fair-E-IPS", "Fair-I-Pers", "Fair-E-Pers"])
    labels = np.asarray(
        ["Naive", "D-ULTR(Glob)", "D-ULTR", "Skyline", "FairCo(Imp)", "FairCo(Exp)", "FairCo(Imp)", "FairCo(Exp)"])
    overall_fairness = get_unfairness_from_rundata(movie_data, pair_group_combinations, trials, n_user)
    ndcg_full = np.asarray([x["NDCG"][:, :n_user] for x in movie_data])
    #colors = ["blue", "orange", "green", "black", "red", "purple", "red", "purple"]
    colors = ["C0", "C1", "C4", "black", "C2", "C3", "C2", "C3"]
    # NDCG plot with Naive, IPS, Pers, Skyline

    ndcg = [ np.mean(ndcg_full[i],axis=0) for i in [0,1,2,3]]
    ndcg_std = [ndcg_full[i] for i in [0,1,2,3]]
    ndcg_labels = labels[[0,1,2,3]]
    ndcg_colors = colors[:4]
    # plt.figure("NDCG", figsize=(15,5))
    for i in range(4):
        plot_ndcg(ndcg[i], ndcg_labels[i], plot=False, window_size=60000, std=ndcg_std[i], color=ndcg_colors[i])
    plt.legend(ncol=2)
    plt.savefig("Movie_data/Paper_Plots/SkylineNDCG.pdf", bbox_inches="tight", dpi=800)
    plt.show()
    plt.close("all")


    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0, 1, 2, 6],
                                     "Movie_data/Paper_Plots/NDCG_UnfairImpactfinal.pdf", 1, colors = colors)

    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0, 1, 2, 7],
                                     "Movie_data/Paper_Plots/NDCG_UnfairExposurefinal.pdf", 0, colors = colors)


    # Unfairness In Impact & Exposure for IPS, Fair-E-Pers, Fair-I-Pers

    selection = [2, 6,7]
    last_colors = [colors[i] for i in selection]
    plot_Exposure_and_Impact_Unfairness(overall_fairness[selection], labels[selection], "Movie_data/Paper_Plots/ImpactVSExposurefinal.pdf", colors=last_colors)

def load_and_plot_news():
    global DATA_SET
    DATA_SET = 0
    trials = 100
    n_user = 3000
    G = [0,1]
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]
    #news_data = np.load("News_data/plots/Random_data_final/Fairness_Data.npy")
    #labels = np.asarray(["Naive", "IPS", "Pers", "Fair-I-IPS", "Fair-I-Pers", "Fair-E-IPS", "Fair-E-Pers"])
    news_data = np.load("News_data/plots/Random_data_final_unconditionalNDCG/Fairness_Data.npy", allow_pickle=True)
    #labels = np.asarray(["Naive", "IPS", "Fair-I-IPS", "Fair-E-IPS"])

    labels = np.asarray(
        ["Naive", "D-ULTR(Glob)", "FairCo(Imp)", "FairCo(Exp)"])
    colors = ["C0", "C1", "C2", "C3"]


    overall_fairness = get_unfairness_from_rundata(news_data, pair_group_combinations, trials, n_user)
    ndcg_full = np.asarray([x["NDCG"][:, :n_user] for x in news_data])

    selection = [0,1,2] #[0, 1, 2, 3, 4]
    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, selection,
                                     "News_data/NDCG_UnfairImpact.pdf", 1, colors=colors)
    selection = [0, 1, 3] # [0, 1, 2, 5, 6]
    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, selection,
                                     "News_data/NDCG_UnfairExposure.pdf", 0,colors=colors)


def load_and_plot_all():
    global DATA_SET
    DATA_SET = True
    skyline_data = np.load("plots/Jokes/Skyline/Fairness_Data.npy")  # Skyline
    exposure_data = np.load("plots/Jokes/Exposure/Fairness_Data.npy")  # Naive Pop, IPS, Fair-E-IPS
    exposure_pers_data = np.load("plots/Jokes/ExposurePers/Fairness_Data.npy")  # Naive Pop, Pers, Fair-E-Pers
    impact_data = np.load("plots/Jokes/ImpactPers/Fairness_Data.npy")  # Fair-I-IPS, Fair-I-Pers

    synthetic_data = np.load("plots/SynteticOverview/Fairness_Data.npy")

    items = [Joke(i) for i in np.arange(0, 90)]
    G = assign_groups(items)
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]

    # NDCG plot with Naive, IPS, Pers, Skyline

    ndcg = [np.mean(exposure_data[0]["NDCG"], axis=0), np.mean(exposure_data[1]["NDCG"], axis=0),
            np.mean(exposure_pers_data[1]["NDCG"], axis=0), np.mean(skyline_data[0]["NDCG"], axis=0)]
    ndcg_std = [exposure_data[0]["NDCG"], exposure_data[1]["NDCG"], exposure_pers_data[1]["NDCG"],
                skyline_data[0]["NDCG"]]
    labels = ["Naive", "IPS", "Pers", "Skyline-Pers"]

    # plt.figure("NDCG", figsize=(15,5))
    for i in range(4):
        plot_ndcg(ndcg[i], labels[i], plot=False, window_size=1, std=ndcg_std[i])
    plt.legend(ncol=2)
    plt.savefig("plots/Paper_Plots/SkylineNDCG.pdf", bbox_inches="tight", dpi=800)
    plt.show()
    plt.close("all")

    # Disparity in Exposure 1. Naive, IPS, Fair-E-IPS, 2. IPS, Pers, Fair-E-Pers

    run_data = [exposure_data[0], exposure_data[1], exposure_data[2], exposure_pers_data[1], exposure_pers_data[2]]
    ndcg_full = [x["NDCG"][:, :7000] for x in run_data]

    labels = ["Naive", "IPS", "Fair-E-IPS", "Pers", "Fair-E-Pers"]

    overall_fairness = get_unfairness_from_rundata(run_data, pair_group_combinations, 10, 7000)
    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0, 1, 3, 2, 4],
                                     "plots/Paper_Plots/NDCG_UnfairExposure.pdf", 0)

    # combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0,1,2],"plots/Paper_Plots/NDCG_UnfairExposurePop.pdf", 0)
    # combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [1,3,4],
    #                                 "plots/Paper_Plots/NDCG_UnfairExposurePers.pdf", 0)

    # Disparity in Impact 1. Naive, IPS, Fair-I-IPS, 2. IPS, Pers, Fair-I-Pers

    run_data = [exposure_data[0], exposure_data[1], impact_data[0], exposure_pers_data[1], impact_data[1]]
    ndcg_full = [x["NDCG"][:, :7000] for x in run_data]
    labels = ["Naive", "IPS", "Fair-I-IPS", "Pers", "Fair-I-Pers"]
    overall_fairness = get_unfairness_from_rundata(run_data, pair_group_combinations, 10, 7000)

    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0, 1, 3, 2, 4],
                                     "plots/Paper_Plots/NDCG_UnfairImpact.pdf", 1)

    # combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0,1,2],"plots/Paper_Plots/NDCG_UnfairImpactPop.pdf", 1)
    # combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [1,3,4],
    #                                 "plots/Paper_Plots/NDCG_UnfairImpactPers.pdf", 1)

    # Unfairness In Impact & Exposure for IPS, Fair-E-Pers, Fair-I-Pers
    run_data = [exposure_data[1], exposure_pers_data[2], impact_data[1]]
    labels = ["IPS", "Fair-E-Pers", "Fair-I-Pers"]
    overall_fairness = get_unfairness_from_rundata(run_data, pair_group_combinations, 10, 7000)
    plot_Exposure_and_Impact_Unfairness(overall_fairness, labels, "plots/Paper_Plots/ImpactVSExposure.pdf")

    # Syntethic Data

    DATA_SET = False
    items = [Item(i) for i in np.linspace(-1, 1, 20)] + [Item(i) for i in np.linspace(-1, 0.01, 10)]
    G = assign_groups(items)
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]
    run_data = synthetic_data
    ndcg_full = [x["NDCG"][:, :5000] for x in run_data]
    labels = ["Naive", "IPS", "Fair-I-IPS", "Fair-E-IPS"]
    overall_fairness = get_unfairness_from_rundata(run_data, pair_group_combinations, 100, 5000)

    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0, 1, 2],
                                     "plots/Paper_Plots/NDCG_UnfairImpactSynthetic.pdf", 1, synthetic=True)
    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0, 1, 2],
                                     "plots/Paper_Plots/NDCG_UnfairExposureSynthetic.pdf", 0, synthetic=True)


def plot_Exposure_and_Impact_Unfairness(overall_fairness, labels, filename, colors=None):
    n = np.shape(overall_fairness)[2]
    plt.figure("Unfairness", figsize=(15, 5))
    fairness_std = np.std(overall_fairness, axis=1)
    overall_fairness = np.mean(overall_fairness, axis=1)
    for i in range(len(labels)):
        plt.subplot(121)
        if colors is None:
            p = plt.plot(np.arange(n), overall_fairness[i, :, 0], label=labels[i])
            color = p[-1].get_color()
        else:
            p = plt.plot(np.arange(n), overall_fairness[i, :, 0], label=labels[i], color=colors[i])
            color = colors[i]
        plt.fill_between(np.arange(n), overall_fairness[i, :, 0] - fairness_std[i, :, 0],
                         overall_fairness[i, :, 0] + fairness_std[i, :, 0], alpha=0.4, color=color)
        ax = plt.gca()
        ax.set_ylim(0, 0.2)
        #ax.set_ylim(0, np.max(overall_fairness[:, :, 0]))
        ax.set_xlim(0, n)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        plt.xlabel("Users")
        plt.ylabel("Exposure Unfairness")
        #ax.legend()

        plt.subplot(122)
        p = plt.plot(np.arange(n), overall_fairness[i, :, 1], label=labels[i], color = color)
        #color = p[-1].get_color()
        plt.fill_between(np.arange(n), overall_fairness[i, :, 1] - fairness_std[i, :, 1],
                         overall_fairness[i, :, 1] + fairness_std[i, :, 1], alpha=0.4, color=color)
        ax = plt.gca()
        ax.set_ylim(0, 0.2)
        #ax.set_ylim(0, np.max(overall_fairness[:, :, 1]))
        ax.set_xlim(0, n)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        plt.xlabel("Users")
        plt.ylabel("Impact Unfairness")
        ax.legend()
    # plt.legend(loc=0)# 'upper right')
    plt.savefig(filename, bbox_inches="tight", dpi=800)

    plt.close("all")


def combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, selection, name, type=0, synthetic=False, colors=None):
    # fig, ax = plt.subplots()
    # ax2 = None
    ax = plt.subplot(121)
    ax2 = plt.subplot(122)
    unfairness_label = "Impact Unfairness" if type == 1 else "Exposure Unfairness"
    for i in selection:
        # ax2 = plot_NDCG_Unfairness(ndcg_full[i], overall_fairness[i, :, :, type], ax=ax, ax2=ax2, label=labels[i],
        plot_NDCG_Unfairness(ndcg_full[i], overall_fairness[i, :, :, type], ax=ax, ax2=ax2, label=labels[i],
                             unfairness_label=unfairness_label, synthetic=synthetic, color=colors[i])

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    order = [0,2,1,3]
    ax2.legend()
    #ax.legend([handles[i] for i in order],[labels[i] for i in order], ncol=2)  # title="NDCG",loc="upper left")
    #ax2.legend([handles2[i] for i in order],[labels2[i] for i in order], ncol=2)  # title="Unfairness", loc ="upper right")
    #ax.set_ylim(np.min(ndcg_full),np.max(ndcg_full))
    #ax.set_ylim(0.8, 1)
    if DATA_SET==2:
        ax.set_ylim(0.7,0.9)
    else:
        ax.set_ylim(0.65, 0.75)
    #ax2.set_ylim(np.min(overall_fairness[:,:,20:,type])*0.95,np.max(overall_fairness[:,:,20:,type])*1.05)
    if type:
        ax2.set_ylim(0, 0.2)
    else:
        ax2.set_ylim(0, 0.4)
    # ax.legend(title="NDCG", loc="center left")
    # ax2.legend(title="Unfairness", loc="center right")

    # plt.legend(ax.lines + ax2.lines, ncol=2)
    # ax.legend(ncol=2)
    # squarify(fig)
    plt.savefig(name, bbox_inches="tight", dpi=800)
    plt.close("all")


def plot_with_errorbar(x, all_stats, methods, filename, x_label, log_x=False, impact=True):
    x = np.asarray(x)

    plt.figure("Errorbar", figsize=(15, 5))
    """
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    """
    ax = plt.subplot(121)
    ax2 = plt.subplot(122)

    if log_x:
        # ax.set_xscale('log')
        ax.set_xscale("symlog", linthreshx=x[0] if x[0] != 0 else x[1])
        ax.set_xticks(x)
        ax2.set_xscale("symlog", linthreshx=x[0] if x[0] != 0 else x[1])
        ax2.set_xticks(x)

    ax.set_xlabel(x_label)
    ax2.set_xlabel(x_label)
    ax.set_ylabel("NDCG")
    #ax.set_ylim([0.4, 1])
    ax.set_ylim([np.min(all_stats[:,:,:,0])-0.01, np.max(all_stats[:,:,:,0])+0.01])
    if impact:
        ax2.set_ylabel("Impact Unfairness")
        # ax.set_ylim([0.95, 1])
        #ax2.set_ylim([0, 0.2])
        ax2.set_ylim([np.min(all_stats[:,:,:,2]) - 0.01, np.max(all_stats[:,:,:,2]) + 0.01])
    else:
        ax2.set_ylabel("Exposure Unfairness")
        #ax2.set_ylim([0, 0.2])
        ax2.set_ylim([np.min(all_stats[:,:,:,1]) - 0.01, np.max(all_stats[:,:,:,1]) + 0.01])
        # ax2.set_ylim([0, 0.2])
        # ax.set_ylim([0.95, 1])
    ax.set_xlim(min(x), max(x))
    ax2.set_xlim(min(x), max(x))
    # ax.set_ylim([0.7, 1])

    for i, m in enumerate(methods):
        print(np.shape(all_stats[:, i, :, 0]),np.shape(x), np.shape(all_stats[:, i, :, 0]))
        ax.errorbar(x * (1. + ((i * 2 - 1) / 100.)), np.mean(all_stats[:, i, :, 0], axis=1),
                    np.std(all_stats[:, i, :, 0], axis=1), label=m)
    for i, m in enumerate(methods):
        if impact:
            ax2.errorbar(x * (1. + ((i * 2 - 1) / 100.)), np.mean(all_stats[:, i, :, 2], axis=1),
                         np.std(all_stats[:, i, :, 2], axis=1), label=m)  # , linestyle=":")
        else:
            ax2.errorbar(x * (1. + ((i * 2 - 1) / 100.)), np.mean(all_stats[:, i, :, 1], axis=1),
                         np.std(all_stats[:, i, :, 1], axis=1), label=m)  # , linestyle=":")

    #ax.legend(loc=0)  # title="NDCG",loc="center left")
    ax2.legend(loc=0)  # title="Unfairness", loc="center right")
    plt.savefig(filename, bbox_inches="tight", dpi=800)
    plt.close("all")


def load_and_plot_lambda_comparison():
    """
    items = [Item(i) for i in np.linspace(-1, 1, 20)] + [Item(i) for i in np.linspace(-1, 0.01, 10)]
    G = assign_groups(items)
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]

    trials, iterations = 10, 2000
    methods = ["Fair-I-IPS-LP", "Fair-I-IPS"]
    n = len(methods)
    data = np.load(
        "plots/LambdaTest/Fairness_Data.npy")  # Fair-I-IPS-LP [0,2,4,6,8]  Fair-I-IPS [1,3,5,7,9] lambda 0.0001, 0.001, 0.01, 0.1, 10
    data100 = np.load("plots/LambdaTest/Fairness_Data100.npy")  # Fair-I-IPS-LP [0]  Fair-I-IPS [1]   lambda 100
    data1 = np.load("plots/LambdaTest/Fairness_Data1.npy")  # Fair-I-IPS-LP [0]  Fair-I-IPS [1]   lambda 1
    data1000 = np.load(
        "plots/LambdaTest/Fairness_Data1000.npy")  # Fair-I-IPS-LP [0]  Fair-I-IPS [1]   lambda 1000, 10000

    #TODO import IPS as baseline?
    data = np.concatenate((data, data1, data100, data1000))
    x = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000,10000 ] # , 1
    data[[8,9,10,11]] = data[[10,11,8,9]]
    plot_lambda_comparison(x, data, methods, trials, pair_group_combinations, "lambdacomparison")


    data = np.load("plots/LambdaTest/Fairness_DataExposure.npy")
    items = [Item(i) for i in np.linspace(-1, 1, 10)] +[Item(i) for i in np.linspace(-1, -0.01, 5)]
    G = assign_groups(items, False)
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]
    x = [0.1, 1, 10, 100, 1000]
    methods = ["Fair-E-IPS-LP", "Fair-E-IPS"]
    trials, iterations = 5, 2000
    plot_lambda_comparison(x, data, methods, trials, pair_group_combinations, "lambdaExposurecomparison")


    data = np.load("plots/LambdaTest/Fairness_DataExposureNoCompensate.npy")
    plot_lambda_comparison(x, data, methods, trials, pair_group_combinations, "lambdaExposureNoCompensate")

    data = np.load("plots/LambdaTest/Fairness_DataAllComparisons.npy")

    methods = ["Fair-I-IPS-LP", "Fair-I-IPS"]
    plot_lambda_comparison(x, data, methods, trials, pair_group_combinations, "lambdaImpactAllComparisonsOld")


    data = np.load("plots/LambdaTest/Fairness_Data_Final.npy")
    """
    data = np.load("News_data/plots/LambdaFinal_3/Fairness_Data.npy", allow_pickle=True)

    #methods = ["Fair-I-IPS-LP", "Fair-I-IPS"]
    methods = ["LinProg", "FairCo(Imp)"]
    pair_group_combinations = [(0,1)]
    trials, iterations = 10, 2000
    x = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

    plot_lambda_comparison(x, data, methods, trials, pair_group_combinations, "lambdaImpactAllComparisons_final_newNews")


def plot_lambda_comparison(x, data, methods, trials, pair_group_combinations, name):
    n = len(methods)
    all_stats = np.zeros((len(data) // 2, len(methods), trials, 3))
    for j in range(len(data)):
        all_stats[j // n, j % n, :, 0] = np.mean(data[j]["NDCG"][:, :], axis=1)

        for a, b in pair_group_combinations:
            all_stats[j // n, j % n, :, 1] += np.abs(data[j]["prop"][:, -1, a] / data[j]["true_rel"][:, -1, a] -
                                                     data[j]["prop"][:, -1, b] / data[j]["true_rel"][:, -1, b])

            all_stats[j // n, j % n, :, 2] += np.abs(data[j]["clicks"][:, -1, a] / data[j]["true_rel"][:, -1, a] -
                                                     data[j]["clicks"][:, -1, b] / data[j]["true_rel"][:, -1, b])
    all_stats[:, :, :, 1:] /= len(pair_group_combinations)
    print("allstats shape", np.shape(all_stats))
    plot_with_errorbar(x, all_stats, methods, "News_data/" + name + ".pdf",
                       r'$\lambda $', log_x=True)
    plot_with_errorbar(x, all_stats, methods,
                       "News_data/" + name + "Exposure.pdf",
                       r'$\lambda $', log_x=True, impact=False)

    plt.close("all")
    for j in range(n):
        plt.scatter((all_stats[:,j,:,0]).flatten(), (all_stats[:,j,:,2]).flatten(),label = methods[j], marker="o")
    plt.xlabel("NDCG")
    plt.ylabel("Impact Unfairness")
    plt.legend()
    plt.savefig("Test.pdf")

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

