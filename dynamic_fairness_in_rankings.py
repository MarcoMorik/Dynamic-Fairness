##Imports
import matplotlib as mpl
"""pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],                    # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
}"""
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
mpl.rcParams.update(params)

mpl.use('Agg')
import numpy as np
from tabulate import tabulate
import scipy.integrate
from enum import Enum  
import matplotlib.pyplot as plt
import scipy.stats
import random
from scipy.stats import gamma
import scipy.special as sps
#from birkhoff import birkhoff
#from sinkhorn_knopp import sinkhorn_knopp as skp
from itertools import combinations 
from itertools import permutations 
from tqdm import tqdm
import pandas as pd
import sys
from IPython.display import clear_output
import time
import warnings; warnings.simplefilter('ignore') ##Ignores Warnings for nicer Plots. Disable for Debugging
# %matplotlib inline
import data_utils
import os
#from birkhoff import birkhoff_von_neumann_decomposition
import birkhoff
import relevance_network
birkhoff.TOLERANCE = 10**(-8)


JOKES = True

"""## Item Class"""

class Item:
    
    def __init__(self,polarity, quality = 1):
        self.p = polarity
        self.q = quality
        
        if(GROUP_BOUNDARIES[0][0]<= polarity <= GROUP_BOUNDARIES[0][1]):
            self.g = 0
        elif(GROUP_BOUNDARIES[1][0]<= polarity <= GROUP_BOUNDARIES[1][1]):
            self.g = 1
        else:
            self.g = 2
    def get_features(self):
        tmp = [0]*3
        tmp[self.g]=1
        #return np.asarray([self.p,self.q, self.p**2] + tmp)
        return np.asarray([self.p,self.q] + tmp)

class Joke:
    def __init__(self, id):
        self.id = id

    def get_features(self):
        return np.asarray([self.id])

"""##Hyperparameter"""

U_ALPHA = 0.5
U_BETA = 0.5
U_STD = 0.3
W_FAIR = 10
KP = 0.001
PROB_W = 5
LP_COMPENSATE_W = 0.025
GROUP_BOUNDARIES = [[-1,-0.33],[0.33,1]] #Boundaries for Left and Right

DATA_FAIR_FULLQUALITY = [Item(x,1) for x in np.linspace(-1,1,27)]
DATA_FAIR_BETTEREXTREME = [Item(x,max(abs(x),0.1)) for x in np.linspace(-1,1,27)]
DATA_FAIR_BETTERLEFT = [Item(x,max(0.1,abs((x-0.2)/1.2))) for x in np.linspace(-1,1,27)]
DATA_FAIR_BETTERLEFT2 = [Item(x,1) if x < -0.33 else Item(x,0.95) for x in np.linspace(-1,1,27)]

DATA_MORELEFT_FULLQUALITY = [Item(x,1) for x in np.linspace(-1,-0.4,15)] +[Item(x,1) for x in np.linspace(-0.3,0.3,5)] +[Item(x,1) for x in np.linspace(0.4,1,5)]
DATA_MORELEFT_BETTERRIGHT = [Item(x,max(0.1,abs((x+0.2)/1.2))) for x in np.linspace(-1,-0.4,15)] +[Item(x,max(0.1,abs((x+0.2)/1.2))) for x in np.linspace(-0.3,0.3,5)] +[Item(x,max(0.1,abs((x+0.2)/1.2))) for x in np.linspace(0.4,1,5)]
DATA_FAIR_DIVERSE = [Item(x,y) for x in np.linspace(-1,1,9) for y in np.linspace(0.11,0.99,3)]

DATASETS = {"FAIR_FULLQUALITY": DATA_FAIR_FULLQUALITY,"FAIR_BETTEREXTREME": DATA_FAIR_BETTEREXTREME,"FAIR_BETTERLEFT":DATA_FAIR_BETTERLEFT, "MORELEFT_FULLQUALITY":DATA_MORELEFT_FULLQUALITY,"MORELEFT_BETTERRIGHT": DATA_MORELEFT_BETTERRIGHT, "FAIR_DIVERSE":DATA_FAIR_DIVERSE}


"""##User Affinity and Distribution"""

#Funktions for User score, position score, assigning groups and  User distributions
def affinity_score(user, items):
    return(affinity_score_adv(user,items))

def affinity_score_adv(user, items):
    if JOKES:
        if (type(items) == list):
            return np.asarray([user[0][x.id] for x in items])
        else:
            return user[0][items.id]
    else:

        #User normal distribution pdf without the normalization factor
        if(type(items) == list):
            item_affs = np.asarray([x.p for x in items])
            item_quality = np.asarray([x.q for x in items])
        else:
            item_affs = items.p
            item_quality = items.q
        aff_prob = np.exp(-(item_affs - user[0])**2 / (2*user[1]**2))*item_quality

        aff_prob = aff_prob
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
    return pos

def assign_groups(items, jokes=JOKES):
    if not jokes:
        G = [[], [], []] # Left, Right, Neutral
        for i, item in enumerate(items):
            G[item.g].append(i)
    else:
        """
        G = [[], [], []]  # Left, Right, Neutral
        for i, item in enumerate(items):
            G[i % 3].append(item.id)
        """
        G = [[i.id] for i in items]
        
    return G

def sample_user_base(distribution = "beta", alpha =U_ALPHA, beta = U_BETA, u_std = U_STD):
    if(distribution == "beta"):
        user = np.random.beta(alpha, beta)
        user *= 2
        user -= 1
        std = u_std
    elif(distribution == "discrete"):
        user = np.random.choice([-1,0,1])
        if(user == 0):
            std = 0.85
        else:
            std = 0.1
    else:
        print("please specify a distribution for the user")
        return (0,1)
    return (user, std)

def sample_user_joke():
    df, features = data_utils.load_data()

    while True:
        for i in range(df.shape[0]):
            yield (df.iloc[i].as_matrix(), features.iloc[i].as_matrix())
        print("All user preferences already given, restarting with the old user!")


def get_ndcg_score(ranking, true_relevances, click_model = "PBM_log"):
    dcg = np.sum(true_relevances[ranking] / np.log2(2+np.arange(len(ranking))))
    idcg = np.sum(np.sort(true_relevances)[::-1] / np.log2(2+np.arange(len(ranking))))
    #dcg = np.sum(true_relevances[ranking] /position_bias(len(ranking),click_model, true_relevances[ranking]))
    #idcg_rank = np.argsort(true_relevances)[::-1]
    #idcg =  np.sum(idcg_rank / position_bias(len(ranking),click_model,true_relevances[idcg_rank]))
    return dcg / idcg

def get_numerical_relevances_base(items, alpha = U_ALPHA, beta = U_BETA, std = U_STD):
    relevances = []
    beta_dist = lambda x: scipy.stats.beta.pdf(x, alpha, beta)
    for item in items:
        aff = lambda x: affinity_score_adv((x,std), item )
        #aff = lambda x: np.exp(-(item - x)**2 / (2*u_std**2))
        rel = lambda x: aff(x) * 0.5* beta_dist((x+1)/2)
        integrated = scipy.integrate.quad(rel,-1,1)
        #print("Integration result", integrated)
        relevances.append(integrated[0])

    return np.asarray(relevances)


def get_numerical_relevances_joke(items):
    df, _ = data_utils.load_data()
    relevances = df.mean().as_matrix()
    #print("Numerical joke relevance: ", relevances)
    return np.asarray(relevances)

if JOKES:
    sample_user_generator = sample_user_joke()
    sample_user = lambda: next(sample_user_generator)
    get_numerical_relevances = lambda x: get_numerical_relevances_joke(x)

else:
    get_numerical_relevances = lambda x: get_numerical_relevances_base(x)
    sample_user = lambda : sample_user_base()

def plot_contour(nn , linear = False, items=None, plot_with_comparison= True):
    p = np.linspace(-1,1,20)
    q = np.linspace(0,1,10)
    xx, yy = np.meshgrid(p,q)
    stacked_features = np.stack((xx.flatten(),yy.flatten()),axis=1)
    if(len(Item(1,1).get_features())==6):
        stacked_features = np.stack((stacked_features[:,0],stacked_features[:,1],stacked_features[:,0]**2,stacked_features[:,0]<-0.33,stacked_features[:,0]>0.33,[-0.33 < x < 0.33 for x in stacked_features[:,0]]), axis=1)
    elif(len(Item(1,1).get_features())==5):
        stacked_features = np.stack((stacked_features[:,0],stacked_features[:,1],stacked_features[:,0]<-0.33,stacked_features[:,0]>0.33,[-0.33 < x < 0.33 for x in stacked_features[:,0]]), axis=1)

    if(linear):
        result = nn.predict(stacked_features, False)
    else:
        result = nn.predict(stacked_features)
    
    if(plot_with_comparison):
        fig, axes = plt.subplots(figsize=(7,3.5),nrows=1, ncols=2, sharey='row')
        axes.flat[0].set_ylabel('Quality')
        tmp_items = [Item(x,1) for x in p]
        relevances_p = get_numerical_relevances(tmp_items)
        relevances = relevances_p[np.newaxis,:] * q[:,np.newaxis]
        z = [relevances,np.reshape(result,np.shape(xx))]
        title = ["True Relevance", "Estimated Relevance"]
        for i, ax in enumerate(axes.flat):
            im = ax.contourf(xx, yy, z[i], vmin=0, vmax=0.4)
            ax.set_xlabel('Polarity')
            ax.set_title(title[i])
            ax.set_xlim(-1,1)
            ax.set_ylim(0,1)
            
        if(items is not None):
            polarities = [x.p  for x in items]
            qualities = [x.q for x in items]
            axes.flat[1].scatter(polarities,qualities,marker='x',color="black")

        fig.subplots_adjust(bottom=0.22)
        cbar_ax = fig.add_axes([0.1, 0, 0.8, 0.05])
        fig.colorbar(im, cax=cbar_ax,orientation='horizontal',boundaries=np.linspace(0, 0.4, 9))
        
    else:
        plt.contourf(xx, yy, np.reshape(result,np.shape(xx)),vmin=0, vmax=0.4)
        plt.xlabel('Polarity')
        plt.ylabel('Quality')
        if(items is not None):
            polarities = [x.p  for x in items]
            qualities = [x.q for x in items]
            plt.scatter(polarities,qualities,marker='x',color="black")
        plt.colorbar(boundaries=np.linspace(0, 0.4, 9),location="bottom")
        #plt.title("Estimated Relevances")
    plt.show()
    
def plot_optimal_contour():
    p = np.linspace(-1,1,20)
    q = np.linspace(0,1,10)
    xx, yy = np.meshgrid(p,q)
    items = [Item(x,1) for x in p]
    relevances_p = get_numerical_relevances(items)
    relevances = relevances_p[np.newaxis,:] * q[:,np.newaxis]
    
    m = plt.contourf(xx, yy, relevances, vmin=0, vmax=0.4)
    plt.xlabel('Polarity')
    plt.ylabel('Quality')
    plt.colorbar( boundaries=np.linspace(0, 0.4, 9),orientation='horizontal')
    plt.title("True Relevances")
    plt.show()

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
    #TODO Probably wrong? The scale has a huge impact on the outcome, that cant be desirable:
    return np.argsort(np.random.gumbel(loc = weighted_popularity))[::-1]

#Rank using a simple P Controller
def controller_rank(weighted_popularity, e_p):
    return np.argsort(weighted_popularity + KP * e_p )[::-1]


#P-L Ranking using softmax
def probabilistic_rank(weighted_popularity):    
    p =np.exp(PROB_W * weighted_popularity)/sum(np.exp(PROB_W *weighted_popularity))
    return np.random.choice(np.arange(len(p)),len(p),replace=False, p=p) 
        
def neural_rank(nn, items, user, joke = JOKES, e_p = 0 ):
    if joke:
        x_test = np.asarray(user[1])
    else:
        x_test = np.asarray(list(map(lambda x: x.get_features(), items)))
    #print("Input  shape", x_test.shape)
    relevances = nn.predict(x_test)
    return np.argsort(relevances+ KP * e_p)[::-1]


#Fair Ranking
def fair_rank(items, popularity,ind_fair=False, group_fair=True, debug=False, w_fair = 1, group_click_rel = None):
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
            if len(G[a]) > 0 and len(G[b])>0 and U[a] >= U[b]: #len(G[a]) * U[a] >= len(G[b]) *U[b]: for comparing mean popularity
                for i in range(n):
                    #tmp1 = 1. / U[a] if i in G[a] else 0 
                    #tmp2 = 1. / U[b] if i in G[b] else 0 
                    tmp1 = popularity[i] / U[a] if i in G[a] else 0 
                    tmp2 = popularity[i] / U[b] if i in G[b] else 0 
                    
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

def ideal_rank(users, item_affs):
    aff_prob = np.zeros(len(item_affs))
    for user in users:
        aff_prob += affinity_score_adv(user, item_affs)
        #aff_prob += scipy.stats.norm.pdf(item_affs, user,0.3)
        #aff_prob /= np.max(aff_prob)
        #aff_prob = np.around(aff_prob, 3)
            
    return aff_prob, (np.argsort(aff_prob)[::-1])

"""##Simulations"""

#CLICK FUNCTION P(click|item) = P(obs|position)*P(relevance|affiliation)*P(click|relevance) = position_bias*affiliation_bias
#rank by expected rel over users P(rel)
def click(user, popularity, items, weighted_popularity=None, G=None, ranking_method="pop", click_model="PBM_log",cum_exposure = None, decomp = None, new_fair_rank = False, nn=None,  integral_fairness = None):
    n = len(popularity)
    click_prob = np.zeros(n)
    fairness_error = None
    
    #Ranking of the entries
    if(ranking_method=="pop"):
        ranking = pop_rank(popularity)
        
    elif(ranking_method=="IPS"):
        assert(weighted_popularity is not None)
        ranking = IPS_rank(weighted_popularity)
        
    elif(ranking_method=="Boost"):
        fairness_boost = get_unfairness(popularity, weighted_popularity, G, error = True, boost=True)
        ranking=boost_rank(weighted_popularity, fairness_boost)
        
    elif(ranking_method=="Exposure_Boost"):
        fairness_boost = get_unfairness(cum_exposure, weighted_popularity, G, error = True, boost=True)
        ranking=boost_rank(weighted_popularity, fairness_boost)
        
    elif(ranking_method=="prob_pop"):
        ranking = probabilistic_rank(popularity)
        
    elif(ranking_method=="prob_IPS"):
        ranking = probabilistic_rank(weighted_popularity)
        
    elif(ranking_method=="prob_Boost"):        
        fairness_boost = get_unfairness(popularity, weighted_popularity, G, error = True, boost=True)
        ranking=probabilistic_rank(weighted_popularity+fairness_boost)
    
    elif("Fair_" in ranking_method or ranking_method=="LP"):
        #Try Linear Programm for fair ranking, when this fails, use last ranking
        if new_fair_rank or decomp is None:
            if(ranking_method=="Fair_IPS"):
                decomp = fair_rank(items, weighted_popularity, debug=False, w_fair=W_FAIR)
            elif(ranking_method=="Fair_Compensate" or ranking_method=="LP"):
                group_fairness = get_unfairness(popularity, weighted_popularity, G, error = False)
                decomp = fair_rank(items, weighted_popularity, debug=False,w_fair=W_FAIR, group_click_rel = group_fairness )
            else:
                raise Exception("Unknown Fair method specified")
            
        if decomp is not None:
            p_birkhoff = np.asarray([np.max([0,x[0]]) for x in decomp])
            p_birkhoff /= np.sum(p_birkhoff)
            sampled_r = np.random.choice(range(len(decomp)), 1, p=p_birkhoff)[0]
            ranking = np.argmax(decomp[sampled_r][1],axis=0)
        else:
            ranking = IPS_rank(weighted_popularity)
            
    elif(ranking_method=="P-Controller"):
        fairness_error = get_unfairness(popularity, weighted_popularity, G, error = True)
        ranking = controller_rank(weighted_popularity,fairness_error)

    elif (ranking_method == "Exposure P-Controller"):
        fairness_error = get_unfairness(cum_exposure, weighted_popularity, G, error=True)
        ranking = controller_rank(weighted_popularity, fairness_error)

    elif(ranking_method=="ind_P-Controller"):
        ind_G = [[i] for i in range(len(popularity))]
        fairness_error = get_unfairness(popularity, weighted_popularity, ind_G, error = True)
        ranking = controller_rank(weighted_popularity,fairness_error)
    elif(ranking_method=="PI-Controller"):
        
        fairness_error = get_unfairness(popularity, weighted_popularity, G, error = True)
        ranking = controller_rank(weighted_popularity,fairness_error + 4* integral_fairness)
    
    elif(ranking_method=="Prob_P-Controller"):
        fairness_error = get_unfairness(popularity, weighted_popularity, G, error = True)
        ranking = probabilistic_rank(weighted_popularity  + KP * fairness_error)
        
    elif(ranking_method=="Random"):
        ranking = random_rank(weighted_popularity)
    
    elif("Neural" in ranking_method):
        if nn is None:
            ranking = IPS_rank(weighted_popularity)
        elif "Exposure-Controller" in ranking_method:
            fairness_error = get_unfairness(cum_exposure, weighted_popularity, G, error=True)
            ranking = neural_rank(nn, items, user, e_p=fairness_error)
        elif "Impact-Controller" in ranking_method:
            fairness_error = get_unfairness(popularity, weighted_popularity, G, error=True)
            ranking = neural_rank(nn, items, user, e_p=fairness_error)

        else:
            ranking = neural_rank(nn, items, user)

    elif(ranking_method == "P-Controll_groupsum"):
        group_clicks = [sum(popularity[G[i]]) for i in range(len(G))]
        group_rel = [max(0.0001,sum(weighted_popularity[G[i]])) for i in range(len(G))]
        fairness_error = np.zeros(n)
        for i in range(len(G)):
            fairness_error[G[i]] = sum([group_clicks[j] for j in range(len(G)) if j != i]) / sum([group_rel[j] for j in range(len(G)) if j != i]) -  group_clicks[i]/ group_rel[i]  
        ranking = controller_rank(weighted_popularity,fairness_error)
    elif(ranking_method == "PI-Controll_groupsum"):
        group_clicks = [sum(popularity[G[i]]) for i in range(len(G))]
        group_rel = [max(0.0001,sum(weighted_popularity[G[i]])) for i in range(len(G))]
        fairness_error = np.zeros(n)
        for i in range(len(G)):
            fairness_error[G[i]] = sum([group_clicks[j] for j in range(len(G)) if j != i]) / sum([group_rel[j] for j in range(len(G)) if j != i]) -  group_clicks[i]/ group_rel[i]  
        ranking = controller_rank(weighted_popularity,fairness_error+ 4* integral_fairness)
    else:
        print("could not find a ranking method called: " + ranking_method)
        raise Exception("No Method specified")
    #create prob of click based on position
    pos = position_bias(n, click_model, weighted_popularity[ranking])
    
    
    #reorder position probabilities to match popularity order
    pos_prob = np.zeros(n)
    pos_prob[ranking] = pos
    
    #get affinity probability based on user and item
    aff_prob = affinity_score_adv(user, items)
    #combine
    #click_prob = pos_prob*aff_prob
    #print(click_prob)
    #Propenenty Score is the examination probablity
    #propensities = 1./ pos_prob
    #return click_prob, propensities, ranking, error
    return aff_prob, pos_prob, ranking, decomp, fairness_error

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
def simulate(popularity, items, ranking_method="pop", click_model="PBM_log", iterations = 2000, numerical_relevance = None):
    G = assign_groups(items)
    weighted_popularity = np.asarray(popularity,dtype = np.float32)
    popularity = np.asarray(popularity)
    pophist = np.zeros((iterations,len(items)))
    w_pophist = np.zeros((iterations,len(items)))
    users = []
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
    nn_errors = []
    
    nn = None

    x_test = np.asarray(list(map(lambda x: x.get_features(), items)))
    
    for i in range(iterations):
        count+=1

        #choose user
        user = sample_user()
        users.append(user)
        relevances += affinity_score_adv(user,items)

        #clicking probabilities
        aff_probs, propensities, ranking, decomp, fairness_error = click(user, popularity, items, weighted_popularity / count, G, ranking_method, click_model, cum_exposure, decomp, count%20 == 9, nn=nn, integral_fairness = cum_fairness_error/count)

        #update popularity
        popularity, weighted_popularity = simulate_click(aff_probs, propensities, popularity, weighted_popularity, ranking, click_model)

        #Save History
        hist[i, :] = ranking
        cum_exposure += propensities
        pophist[i, :] = popularity
        if "Neural" in ranking_method and i > 99 and not JOKES:
            w_pophist[i, :] = predicted_relevances * count
        else:
            w_pophist[i, :] = weighted_popularity

        #update neural network
                    
        if "Neural" in ranking_method:
            if(i == 99): # Initialize Neural Network

                if JOKES:
                    #Train Simple Net (no hiddenlayers) on Joke Dataset
                    train_x = np.asarray([u[1] for u in users])
                    if not "baseline" in ranking_method:
                        nn = relevance_network.relevance_estimating_network(len(user[1]),output_dim=len(items), hidden_units=0, joke= True)
                        train_y = w_pophist[:i+1] - np.concatenate((np.zeros((1,len(items))),w_pophist[:i]))
                        print("The Joke Network is trained with x of shape", np.shape(train_x), "and y of shape", np.shape(train_y))

                    else:
                        #Supervised Baseline
                        nn = relevance_network.relevance_estimating_network(len(user[1]), output_dim=len(items),
                                                                            hidden_units=0, joke=True, supervised=True)
                        train_y = np.asarray([u[0] for u in users])
                        print("The Joke Network is trained with x of shape", np.shape(train_x), "and y of shape",
                              np.shape(train_y))
                    nn.train(train_x, train_y, epochs=600 )

                elif (ranking_method == "Neural_IPS" or ranking_method=="Neural Impact-Controller" or ranking_method =="Neural Exposure-Controller"):
                        nn=relevance_network.relevance_estimating_network(len(items[0].get_features()))
                        clicked = popularity >=0 #Items clicked at least twice
                        nn.train(x_test[clicked],np.clip(weighted_popularity[clicked]/count,0,1), epochs=5000)

                elif(ranking_method=="Linear_Neural"):
                    nn=relevance_network.linear_one_hot_network(len(items[0].get_features()),len(items))

                    clicked = [i for i in range(len(popularity))  if popularity[i] >=3] #Items clicked at least twice
                    nn.train(x_test,np.clip(weighted_popularity/count,0,1), epochs = 5000, consider= clicked)
                    #nn.train(x_test,np.clip(weighted_popularity/count,0,1), epochs = 400, consider= clicked)
                if JOKES:
                    print(user[1])
                    print(np.shape(user),np.shape(user[0]),np.shape(user[1]))
                    predicted_relevances = nn.predict(user[1])
                    #print("Neural Network error:", np.linalg.norm(predicted_relevances - user[0]))
                    nn_errors.append(np.mean((predicted_relevances - user[0])**2))
                else:
                    predicted_relevances = nn.predict(x_test)
                    nn_errors.append(np.mean((predicted_relevances - relevances/(i+1))**2))

            elif(i >99 and i %10 == 9):
                if JOKES:
                    if "baseline" in ranking_method:
                        train_y = np.concatenate((train_y, np.asarray([u[0] for u in users[-10:]])))
                    else:
                        train_y = np.concatenate((train_y, w_pophist[i-9:i+1] - w_pophist[i-10:i] ))
                    train_x = np.concatenate((train_x, np.asarray([u[1] for u in users[-10:]]) ))
                    nn.train(train_x, train_y, epochs=100)
                    predicted_relevances = nn.predict(user[1])
                    #print("Neural Network error:", np.linalg.norm(predicted_relevances - user[0]))

                else:
                    clicked = np.arange(len(items))
                    if(ranking_method =="Neural_IPS"):
                        nn.train(x_test[clicked],np.clip(weighted_popularity[clicked]/count,0,1), epochs = 400)
                    else:
                         nn.train(x_test,np.clip(weighted_popularity/count,0,1), epochs = 400, consider= clicked)
                    predicted_relevances = nn.predict(x_test)
            if JOKES and i >= 99:
                nn_errors.append(np.mean((predicted_relevances - user[0])**2))
            elif i>= 99:
                nn_errors.append(np.mean((predicted_relevances - relevances/(i+1))**2))
            if(i %100 == 99 and False):
                plot_contour(nn, linear= ranking_method=="Linear_Neural", items=items)

        
        #Save statistics
        if(fairness_error is not None):
            cum_fairness_error += fairness_error
            

        NDCG[i] = get_ndcg_score(ranking, relevances/count) #numerical_relevance)
        #NDCG[i] = get_ndcg_score(ranking, relevances/count)
        #print( np.sum(np.abs(numerical_relevance - (relevances/count))))
        
        group_prop[i,:] = [sum(cum_exposure[G[i]]) for i in range(len(G))]
        group_clicks[i,:] =  [sum(popularity[G[i]]) for i in range(len(G))]
        #group_rel[i,:] = [sum(weighted_popularity[G[i]])/count for i in range(len(G))] #Normalizing by Count to have time normalized relevances
        #true_group_rel[i,:] = [sum(relevances[G[i]])/count for i in range(len(G))]
        group_rel[i,:] = [sum(weighted_popularity[G[i]]) for i in range(len(G))] # Having the sum of weighted clicks. Nicer Plots for the fairness measure
        #true_group_rel[i,:] = [sum(relevances[G[i]]) for i in range(len(G))]
        true_group_rel[i,:] = [sum(numerical_relevance[G[i]])*count for i in range(len(G))] # Try to avoid the spikes in the beginning
        
        
        prev = np.copy(ranking)
    if("Neural" in ranking_method and not JOKES):
        plot_contour(nn , linear= ranking_method=="Linear_Neural", items=items)
        #plot_contour(nn,linear= ranking_method=="Linear_Neural")
    #print( numerical_relevance ,"\n",(relevances/count) )
    #rank by expected rel over users P(rel)
    #calculate "ideal" rank
    #users = list of "affiliation scores"
    ideal_vals, ideal_ranking = ideal_rank(users,items)
    
    mean_relevances = relevances / count
    mean_exposure = cum_exposure / count
    
    #true_group_rel[:10,:]=true_group_rel[10,:][np.newaxis,:] 
    fairness_hist = {"prop": group_prop, "clicks": group_clicks, "rel": group_rel, "true_rel": true_group_rel, "NDCG": NDCG}
    return count, hist, pophist, ranking, users, ideal_ranking, mean_relevances, w_pophist, nn_errors, mean_exposure, fairness_hist

def simulate_click(aff_probs, propensities, popularity, weighted_popularity, ranking, click_model):
    if "PBM" in click_model:
        rand_var = np.random.rand(len(aff_probs))
        #clicks = rand_var < (aff_probs * propensities)
        trust_bias = 1 # np.random.normal(np.linspace(0.8,1.2,len(propensities)),0.1,len(propensities))
        clicks = rand_var < ((aff_probs * propensities) * trust_bias)
        popularity += clicks
        weighted_popularity += clicks / propensities  
    else:
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
    return popularity, weighted_popularity

"""#Testing

##Testing function
"""
def test_ranking(items, start_popularity, trials = 100, methods = ["pop","IPS"], click_models = ["PBM_log"], save=False, iterations = 2000):
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
                iterations, ranking_hist, popularity_hist, final_ranking, users, ideal, mean_relevances, w_pophist, errors, mean_exposure, fairness_hist = \
                simulate(popularity, items, ranking_method=method, click_model=click_model, iterations=iterations, numerical_relevance = numerical_relevance)
                iters.append(iterations)
                for r in final_ranking[:3]:
                    for g in range(len(G)):
                        top[g] += 1 if r in G[g] else 0
                        
                rel += mean_relevances
                clicks += popularity_hist[-1]
                exposure += mean_exposure
                trial_NDCG += fairness_hist["NDCG"]
                if("pop" in method):
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
    
    
    labels = [ a + "\n" + b for a in click_models for b in methods]
    
    plot_click_bar_plot(frac_c,labels,save)
    
    for i in range(len(NDCG)):
        plot_ndcg(NDCG[i],labels[i],plot=False)
    plt.legend()
    plt.show()
    return result
    #grouped_bar_plot(statistics, G, [ a + "\n" + b for a in click_models for b in methods])

#Function that simulates and monitor the convergence to the relevance + the developement of cummulative fairness
def collect_relevance_convergence(items, start_popularity, trials = 100, methods = ["pop","IPS"], click_models = ["PBM_log"], iterations = 2000, plot_individual_fairness = True):
    rel_diff = []
    G = assign_groups(items)
    overall_fairness = np.zeros((len(click_models)*len(methods),iterations,8))
    pair_group_combinations = [(a,b) for a in range(len(G)) for b in range(a+1,len(G))]
    count=0
    NDCG = []
    frac_c = [[] for i in range(len(G))]
    nn_errors = []
    for click_model in click_models:
        for method in methods:
            start_time = time.time()
            rel_diff_trial = []
            rel_diff_top20 = []
            fairness = {"prop": np.zeros((iterations,len(G))), "clicks": np.zeros((iterations,len(G))), "rel": np.zeros((iterations,len(G))), "true_rel": np.zeros((iterations,len(G))), "NDCG": np.zeros(iterations)}
            for i in range(trials):
                popularity = np.copy(start_popularity)
                
                #Run Simulation
                iterations, ranking_hist, popularity_hist, final_ranking, users, ideal, mean_relevances, w_pophist, errors, mean_exposure, fairness_hist = \
                simulate(popularity, items, ranking_method=method, click_model=click_model, iterations = iterations)
                ranking_hist = ranking_hist.astype(int)
                if "Neural" in method:
                    nn_errors.append(errors)
                #Calculate the relevance difference between true relevance and approximation
                if(method =="pop"):
                    rel_diff_trial.append(np.mean(np.abs(popularity_hist / (np.arange(iterations)+1)[:,np.newaxis] - (mean_relevances)[np.newaxis,:]),axis=1))
                    if(len(items)>20):
                        before = np.abs(popularity_hist[np.arange(popularity_hist.shape[0])[:,np.newaxis],ranking_hist[:,:20]]/ (np.arange(iterations)+1)[:,np.newaxis] - (mean_relevances[ranking_hist[:,:20]]))
                        cur_estimate = np.mean(before, axis=1)
                        rel_diff_top20.append(cur_estimate)    
                else:
                    rel_diff_trial.append(np.mean(np.abs(w_pophist / (np.arange(iterations)+1)[:,np.newaxis] - (mean_relevances)[np.newaxis,:]),axis=1))
                    if(len(items)>20):
                        rel_diff_top20.append(np.mean(np.abs(w_pophist[np.arange(popularity_hist.shape[0])[:,np.newaxis],ranking_hist[:,:20]] / (np.arange(iterations)+1)[:,np.newaxis] - (mean_relevances[ranking_hist[:,:20]])), axis=1))
                
                for a,b in pair_group_combinations:
                    overall_fairness[count,:,0] += np.abs(fairness_hist["prop"][:,a] / fairness_hist["rel"][:,a] - fairness_hist["prop"][:,b] / fairness_hist["rel"][:,b])
                    overall_fairness[count,:,1] += np.abs(fairness_hist["prop"][:,a] / fairness_hist["true_rel"][:,a] - fairness_hist["prop"][:,b] / fairness_hist["true_rel"][:,b])
                    overall_fairness[count,:,2] += np.abs(fairness_hist["clicks"][:,a] / fairness_hist["rel"][:,a] - fairness_hist["clicks"][:,b] / fairness_hist["rel"][:,b])
                    overall_fairness[count,:,3] += np.abs(fairness_hist["clicks"][:,a] / fairness_hist["true_rel"][:,a] - fairness_hist["clicks"][:,b] / fairness_hist["true_rel"][:,b])
                    
                    greater_estimate = 2*(fairness_hist["rel"][:,a] >  fairness_hist["rel"][:,b]) -1
                    greater_true = 2*(fairness_hist["true_rel"][:,a] >  fairness_hist["true_rel"][:,b]) -1
                    
                    overall_fairness[count,:,4] += np.clip(greater_estimate* (fairness_hist["prop"][:,a] / fairness_hist["rel"][:,a] - fairness_hist["prop"][:,b] / fairness_hist["rel"][:,b]),a_min =0, a_max=None)
                    overall_fairness[count,:,5] += np.clip(greater_true* (fairness_hist["prop"][:,a] / fairness_hist["true_rel"][:,a] - fairness_hist["prop"][:,b] / fairness_hist["true_rel"][:,b]),a_min =0, a_max=None)
                    overall_fairness[count,:,6] += np.clip(greater_estimate* (fairness_hist["clicks"][:,a] / fairness_hist["rel"][:,a] - fairness_hist["clicks"][:,b] / fairness_hist["rel"][:,b]),a_min =0, a_max=None)
                    overall_fairness[count,:,7] += np.clip(greater_true* (fairness_hist["clicks"][:,a] / fairness_hist["true_rel"][:,a] - fairness_hist["clicks"][:,b] / fairness_hist["true_rel"][:,b]),a_min =0, a_max=None)
                    
                
                #Cummulative Fairness per Iteration summed over trials     
                for key, value in fairness_hist.items():
                    fairness[key] += value
                if(trials <=1):
                    #Plot Group Clicks and Items Average Rank
                    group_item_clicks(popularity_hist[-1],G)
                    plot_average_rank(ranking_hist,G)
                    #print("Real Relevances: \n ", np.round(mean_relevances,3))
                    #print("Estimated Relevances: \n", np.round(w_pophist[-1]/(iterations+1),3))
                    if method =="pop":
                        print("Relevance Difference: ", np.sum((mean_relevances - popularity_hist[-1] / (iterations + 1)) ** 2))
                    else:
                        print("Relevance Difference: ", np.sum((mean_relevances -w_pophist[-1]/(iterations+1))**2))
                    #Plot Ranking History
                    plt.title("Ranking History")
                    plt.axis([0, iterations, 0,len(items)])
                    if len(G) <=3:
                        group_colors = {0:"blue",1:"red",2:"black"}
                    else:
                        group_colors = [None for i in range(len(G))]
                    item_rank_path = np.ones((iterations,len(items)))
                    #print("Debuging ranking plot", ranking_hist[:3,:])
                                      
                    for i in range(iterations):
                        item_rank_path[i,ranking_hist[i,:]]=np.arange(len(items))
                    for i in range(len(items)):
                        group_color_i = group_colors[[x for x in range(len(G)) if i in G[x]][0]]
                        plt.plot(np.arange(iterations),item_rank_path[:,i], color=group_color_i)
                    #plt.show()
                    plt.savefig("Rankinghistory_"+click_model+"_"+method+".png")
            print("Time for " + click_model + " " + method+ " was: {0:.4f}".format(time.time()-start_time))        
            #Normalize
            for key, value in fairness.items():
                    fairness[key] /=  trials
            overall_fairness[count,:,:] /= trials
            
            count += 1
            #Plot the Fairness per Group for a single model
            if(plot_individual_fairness):
                plot_fairness_over_time(fairness, G,  click_model.replace("_log","") + " "+ method)
            else:
                #clear_output(wait=True)
                pass
            #Collect Data for later
            NDCG.append(fairness["NDCG"])
            for i in range(len(G)):
                frac_c[i].append(fairness["clicks"][-1,i] / iterations)
                
            if(len(rel_diff_trial)==1):
                rel_tmp = np.asarray(rel_diff_trial[0])
            else:
                rel_tmp = np.mean(np.asarray(rel_diff_trial),axis=0)
            rel_diff.append([rel_tmp,click_model.replace("_log","") + " "+ method])
            if(len(items)>20 and False):
                rel_diff.append((np.mean(np.asarray(rel_diff_top20),axis=0),click_model.replace("_log","") + " "+ method + "top 20"))
    
    #Plot NDCG
    plt.figure("NDCG")
    #plt.title("Average NDCG")
    #labels = [ a + "\n" + b for a in click_models for b in methods]
    labels = [ b for a in click_models for b in methods]
    for i, nd in enumerate(NDCG):
        plot_ndcg(nd,label=labels[i], plot=False)
    plt.legend()
    ax= plt.gca()
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))

    plt.savefig("NDCG.png", bbox_inches="tight", dpi=800)
    #files.download("NDCG.png") 

    plt.show()
    plt.close("all")
    
    #Plot Clicks
    plot_click_bar_plot(frac_c,labels, save=True)
    
    if True:

        plt.close("all")
        #Plot Convergence of Relevance
        for y in rel_diff:
            plt.plot(np.arange(len(y[0])),y[0], label=y[1])
        plt.legend()
        plt.axis([0,len(y[0]),0,0.4])
        plt.ylabel("Avg diff between \n True & Estimated Relevance  ")
        plt.xlabel("Iterations")
        plt.savefig("Relevance_convergence.png")
        plt.show()

    plot_neural_error(nn_errors,  [b for a in click_models for b in methods if "Neural" in b])
    #Plot Unfairness over time between different models
    plot_unfairness_over_time(overall_fairness, click_models, methods)
    #plot_unfairness_over_time(overall_fairness[:,:,4:], click_models, methods)
    

def plot_neural_error(errors, labels):
    plt.close("all")
    # Plot Neural error
    for i, error in enumerate(errors):
        plt.plot(np.arange(100, len(error)+100), error, label=labels[i])
    plt.legend()
    plt.ylabel("Difference True and estimated relevance")
    plt.xlabel("Iterations")
    plt.savefig("Neural_Error.png")

def plot_click_bar_plot(frac_c,labels, save=False):
    group_colors = {-1:"blue",1:"red",0:"black"}
    n = len(labels)
    plt.bar(np.arange(n), frac_c[0], color=group_colors[-1], edgecolor='white', width=1, label="Negative")
    plt.bar(np.arange(n), frac_c[1], bottom=frac_c[0], color=group_colors[1], edgecolor='white', width=1, label="Positive")
    plt.bar(np.arange(n), frac_c[2], bottom=np.add(frac_c[0],frac_c[1]), color=group_colors[0], edgecolor='white', width=1, label="Neutral")
    plt.xticks(np.arange(n), labels , fontweight='bold')
    total_clicks = np.round(np.add(np.add(frac_c[0],frac_c[1]),frac_c[2]),3)
    for i in range(n):
        plt.text(x =i  , y = total_clicks[i] , s = total_clicks[i], size = 10)
    plt.ylabel("Average number of clicks")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if(save):
        plt.savefig("Barplot.png")
    else:
        plt.show()
    
    
def grouped_bar_plot(statistics, groups, models):    
    
    # set width of bar
    barWidth = 0.25
    for j in range(len(models)):
        x_pos = np.arange(3.)
        colors = ["black","grey","orange"]
        i=0
        for key, value in statistics.items(): 
            v = np.asarray(value[j])

            plt.bar(x_pos, [sum(v[groups[i]]) for i in range(len(groups))], color=colors[i], width=barWidth, edgecolor='white', label=key)
            x_pos += barWidth
            i +=1
        # Add xticks on the middle of the group bars
        plt.xlabel('Group', fontweight='bold')
        plt.xticks([r + barWidth for r in range(3)], ["Left","Right","Neutral"])
        plt.title(models[j])
        # Create legend & Show graphic
        plt.legend()
        plt.show()
        #plt.savefig("Proportionalitybar"+str(j)+ ".png")
        #files.download("Proportionalitybar"+str(j)+ ".png") 
        plt.close("all")
        
#Plot that shows the clicks per item within each group
def group_item_clicks(clicks, G):
    if len(G) <=3 :
        group_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
        color_dict = {0: "blue", 1: "red", 2: "black"}

        for i, g in enumerate(G):
            plt.plot(np.arange(len(g)), np.sort(clicks[g])[::-1], color=color_dict[i], label=group_dict[i], marker='o')
    else:

        for i, g in enumerate(G):
            plt.plot(np.arange(len(g)), np.sort(clicks[g])[::-1], label = i,  marker='o')
    plt.xlabel("Item")
    plt.ylabel("Clicks")
    plt.title("Clicks per Item and Group")
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    plt.close("all")
        
#Plot the average rank per item
def plot_average_rank(ranking_hist, G):
    if len(G) <=3:
        group_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
        color_dict = {0: "blue", 1: "red", 2: "black"}
    else:
        group_dict = dict([(i,x) for i, x in enumerate(G)])
        colors = None
    iterations, n = np.shape(ranking_hist)
    summed_ranks = np.zeros(n)
    for i in range(iterations):
        summed_ranks[ranking_hist[i]] += np.arange(1,n+1)
    avg_rank = summed_ranks / iterations
    group_per_item = np.ones(n) * -1
    for i in range(len(G)):
        group_per_item[G[i]] = i
    labels = [group_dict[x] for x in group_per_item]
    if len(G) <= 3:
        colors = [color_dict[x] for x in group_per_item]
    plt.bar(np.arange(n),avg_rank, color=colors)
    plt.xlabel("Items")
    plt.ylabel("Average Rank")
    plt.title("Average Rank per Item")
    plt.show()
    plt.close("all")
    
#Plot Unfairness (Difference between groups) over iterations    
def plot_unfairness_over_time(overall_fairness, click_models, methods):
    n = np.shape(overall_fairness)[1]
    #plt.figure("Unfairness",figsize=(16,4))
    plt.figure("Unfairness",figsize=(17,4))
    
    for i in range(len(click_models)*len(methods)):
        plt.subplot(141)
        #plt.title("Group-Difference \n Exposure per est. Relevance")
        #plt.axis([0,n,0,3])
        plt.plot(np.arange(n), overall_fairness[i,:,0], label=methods[i%len(methods)] +" " + click_models[i//len(methods)])
        plt.xlabel("Iterations")
        plt.ylabel("Group-Difference of \n Exposure per est. Relevance")
        ax= plt.gca()
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        
        plt.subplot(142)
        #plt.axis([0,n,0,1])
        #plt.title("Group-Difference \n Exposure per True Relevance")
        plt.plot(np.arange(n), overall_fairness[i,:,1], label=methods[i%len(methods)] +" " + click_models[i//len(methods)])
        ax= plt.gca()
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))
        plt.xlabel("Iterations")
        plt.ylabel("Group-Difference of \n Exposure per True Relevance")
        
        plt.subplot(143)        
        #plt.axis([0,n,0,1])
        #plt.title("Group-Difference \n Clicks per est. Relevance")
        plt.plot(np.arange(n), overall_fairness[i,:,2], label=methods[i%len(methods)] +" " + click_models[i//len(methods)])
        plt.xlabel("Iterations")
        plt.ylabel("Group-Difference of \n Clicks per est. Relevance")
        ax= plt.gca()
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        
        plt.subplot(144)
        #plt.title("Group-Difference \n Clicks per True Relevance")
        plt.plot(np.arange(n), overall_fairness[i,:,3], label=methods[i%len(methods)] +" " + click_models[i//len(methods)])
        
        #plt.axis([0,n,0,1])
        ax= plt.gca()
        #ax.set_ylim(0,1)
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))
        plt.xlabel("Iterations")
        plt.ylabel("Group-Difference of \n Clicks per True Relevance")
       
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("Unfairness.png", bbox_inches="tight", dpi=800)
    #files.download("Unfairness.png") 
    
    plt.show("Unfairness")
    
    
def plot_fairness_over_time(fairness, G, model):
    n = np.shape(fairness["rel"])[0]
    group_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
    normalize_iter = np.arange(n)
    normalize_iter[0] = 1#Zerost iteration should be divided by 1
    
    """
    plt.figure(figsize=(12,4))
    for g in range(len(G)):
        plt.subplot(131+g)
        if "DCM" in model:
            plt.axis([0,n,0,0.7])
        else:
            plt.axis([0,n,0,2])
        if "pop" in model:
            fairness["rel"] = fairness["clicks"]
        
        #plt.plot(np.arange(n), fairness["rel"][:,g] / normalize_iter /len(G[g]) , label="Estimated Relevance")
        #plt.plot(np.arange(n), fairness["true_rel"][:,g] / normalize_iter /len(G[g]) , label="True Relevance")
       
        
        #plt.plot(np.arange(n), fairness["clicks"][:,g] / normalize_iter , label="Clicks")
        #plt.plot(np.arange(n), fairness["prop"][:,g] / normalize_iter /len(G[g]) , label="Propensities")

        plt.plot(np.arange(n), fairness["prop"][:,g] / fairness["rel"][:,g], label="Exposure per est. Relevance")
        plt.plot(np.arange(n), fairness["prop"][:,g] / fairness["true_rel"][:,g], label="Exposure per True Relevance")
        
        plt.plot(np.arange(n), fairness["clicks"][:,g] / fairness["rel"][:,g], label="Clicks per est. Relevance")
        plt.plot(np.arange(n), fairness["clicks"][:,g] / fairness["true_rel"][:,g], label="Clicks per True Relevance")
        
        plt.xlabel("Iterations")
        plt.title("Model "+ model + "\n Group " + group_dict[g])
        
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    """
    
    color_dict = {0: "blue", 1: "red", 2: "black"}
    plt.figure("Fairness",figsize=(17,4))
    for g in range(len(G)):
        plt.subplot(141)
        #plt.axis([0,n,0,3])
        plt.title("Model "+ model + "\n Exposure per est. Relevance")
        plt.plot(np.arange(n), fairness["prop"][:,g] / fairness["rel"][:,g], label="Group " + group_dict[g], color=color_dict[g])
        plt.subplot(142)
        plt.axis([0,n,0,1])
        plt.title("Model "+ model + "\n Exposure per True Relevance")
        plt.plot(np.arange(n), fairness["prop"][:,g] / fairness["true_rel"][:,g], label="Group " + group_dict[g], color=color_dict[g])
        plt.subplot(143)
        #plt.axis([0,n,0,1])
        plt.title("Model "+ model + "\n Clicks per est. Relevance")
        plt.plot(np.arange(n), fairness["clicks"][:,g] / fairness["rel"][:,g], label="Group " + group_dict[g], color=color_dict[g] )
        plt.subplot(144)
        plt.title("Model "+ model + "\n Clicks per True Relevance")
        plt.plot(np.arange(n), fairness["clicks"][:,g] / fairness["true_rel"][:,g], label="Group " + group_dict[g], color=color_dict[g])
        plt.axis([0,n,0,1])
                  
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig("Fairness"+ model+ ".png")
    plt.show()
    #files.download("Fairness"+ model+ ".png") 
    
def plot_ndcg(ndcg, label="", plot=True, figure_name="NDCG"):
    plt.figure(figure_name)
    plt.plot(np.arange(len(ndcg)),ndcg, label = label)
    plt.axis([0,len(ndcg),0.8,1])
    plt.xlabel("Iterations")
    plt.ylabel("NDCG")
    if(plot):
        plt.title("Average NDCG in model " + label)
        plt.show()
        
def plot_numerical_relevance(items, alpha = U_ALPHA, beta = U_BETA, u_std = U_STD):
    relevances = []
    beta_dist = lambda x: scipy.stats.beta.pdf(x, alpha, beta)
    for item in items:
        aff = lambda x: affinity_score_adv((x,u_std), item )
        #aff = lambda x: np.exp(-(item - x)**2 / (2*u_std**2))
        rel = lambda x: aff(x) * 0.5* beta_dist((x+1)/2)
        integrated = scipy.integrate.quad(rel,-1,1)
        #print("Integration result", integrated)
        relevances.append(integrated)
    acc = 25
    plot_relevances = np.zeros(acc)    
    for i,j in enumerate(np.linspace(-1,1,acc)):
        aff = lambda x: affinity_score_adv((x,u_std), j )
        rel = lambda x: aff(x) * 0.5* beta_dist((x+1)/2)                
        plot_relevances[i] =  scipy.integrate.quad(rel,-1,1)[0]
    plt.plot(np.linspace(-1,1,acc),plot_relevances)
    plt.axis([-1,1,0,0.5])
    plt.ylabel("Relevance")
    plt.xlabel("Documents Affinity")
    plt.show()
    return np.asarray(relevances)

PLOT_PREFIX = "plots/"
def __main__():
    global PLOT_PREFIX
    items = [ Joke(i) for i in np.arange(0,90)]
    popularity = np.zeros(len(items))
    trials = 10

    #overview = test_ranking(items, popularity, trials, click_models=["PBM_log"], methods=["pop", "IPS", "P-Controller"], save = True, iterations=5000)
    #print(tabulate(overview, headers='keys', tablefmt='psql'))


    PLOT_PREFIX = "plots/Exposure/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)
    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                  methods=["pop", "IPS", "Exposure P-Controller", "Neural Exposure-Controller"], iterations=5000, plot_individual_fairness = False)

    #TODO Baseline Probably broken

    PLOT_PREFIX = "plots/Impact/"
    if not os.path.exists(PLOT_PREFIX):
        os.makedirs(PLOT_PREFIX)

    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                             methods=["pop", "IPS", "P-Controller", "Neural Impact-Controller"], iterations=5000, plot_individual_fairness = False)


__main__()