import numpy as np
import scipy.integrate
import scipy.stats
from matplotlib import pyplot as plt
import matplotlib as mpl

from config import ex
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}


def init_plotting():
    mpl.rcParams.update(params)
    mpl.use('Agg')
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





#TODO, should get called not all functions...
from dynamic_fairness_in_rankings import Item, get_numerical_relevances, PLOT_PREFIX, U_ALPHA, U_BETA, U_STD, \
    affinity_score_adv, Joke, assign_groups, get_unfairness_from_rundata


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


def plot_neural_error(errors, labels):
    plt.close("all")
    # Plot Neural error
    for i, error in enumerate(errors):
        plt.plot(np.arange(100, len(error)+100), error, label=labels[i])
    plt.legend()
    plt.ylabel("Difference True and estimated relevance")
    plt.xlabel("Users")
    plt.savefig(PLOT_PREFIX + "Neural_Error.pdf", bbox_inches="tight")


def plot_click_bar_plot(frac_c,labels, save=False):
    group_colors = {-1:"blue",1:"red",0:"black"}
    n = len(labels)
    plt.bar(np.arange(n), frac_c[0], color=group_colors[-1], edgecolor='white', width=1, label="Negative")
    plt.bar(np.arange(n), frac_c[1], bottom=frac_c[0], color=group_colors[1], edgecolor='white', width=1, label="Positive")
    if len(frac_c) >2:
        plt.bar(np.arange(n), frac_c[2], bottom=np.add(frac_c[0],frac_c[1]), color=group_colors[0], edgecolor='white', width=1, label="Neutral")
    plt.xticks(np.arange(n), labels , fontweight='bold')
    total_clicks = np.round(np.sum(np.asarray(frac_c),axis=0),3) # np.round(np.add(np.add(frac_c[0],frac_c[1]),frac_c[2]),3)
    for i in range(n):
        plt.text(x =i  , y = total_clicks[i] , s = total_clicks[i], size = 10)
    plt.ylabel("Average number of clicks")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if(save):
        plt.savefig(PLOT_PREFIX + "Barplot.pdf", bbox_inches="tight")
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
        #plt.savefig(PLOT_PREFIX + "Proportionalitybar"+str(j)+ ".pdf")
        #files.download("Proportionalitybar"+str(j)+ ".pdf")
        plt.close("all")


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


def plot_unfairness_over_time(overall_fairness, click_models, methods, only_two = True):
    n = np.shape(overall_fairness)[2]
    #plt.figure("Unfairness",figsize=(16,4))
    if(only_two):
        plt.figure("Unfairness",figsize=(10,4))
    else:
        plt.figure("Unfairness",figsize=(17,4))
    fairness_std = np.std(overall_fairness, axis=1)
    overall_fairness = np.mean(overall_fairness, axis=1)
    for i in range(len(click_models)*len(methods)):
        if not only_two:
            plt.subplot(141)
            #plt.title("Group-Difference \n Exposure per est. Relevance")
            #plt.axis([0,n,0,3])
            p = plt.plot(np.arange(n), overall_fairness[i,:,0], label=methods[i%len(methods)] )#+" " + click_models[i//len(methods)])
            color = p[-1].get_color()
            plt.fill_between(np.arange(n), overall_fairness[i,:,0]- fairness_std[i,:,0], overall_fairness[i,:,0] +  fairness_std[i,:,0], alpha=0.4, color=color)

            plt.xlabel("Users")
            plt.ylabel("Group-Difference of \n Exposure per est. Relevance")
            ax= plt.gca()
            x0,x1 = ax.get_xlim()
            y0,y1 = ax.get_ylim()
        if only_two:
            plt.subplot(121)
        else:
            plt.subplot(142)
        #plt.axis([0,n,0,1])
        #plt.title("Group-Difference \n Exposure per True Relevance")
        p=plt.plot(np.arange(n), overall_fairness[i,:,1], label=methods[i%len(methods)] )#+" " + click_models[i//len(methods)])
        color = p[-1].get_color()
        plt.fill_between(np.arange(n), overall_fairness[i, :, 1] - fairness_std[i, :, 1],
                         overall_fairness[i, :, 1] + fairness_std[i, :, 1], alpha=0.4, color=color)
        ax= plt.gca()
        ax.set_ylim(0, np.max(overall_fairness[i,15:,1]))
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))
        plt.xlabel("Users")
        plt.ylabel("Group-Difference of \n Exposure per True Relevance")
        if not only_two:
            plt.subplot(143)
            #plt.axis([0,n,0,1])
            #plt.title("Group-Difference \n Clicks per est. Relevance")
            p = plt.plot(np.arange(n), overall_fairness[i,:,2], label=methods[i%len(methods)])# +" " + click_models[i//len(methods)])
            color = p[-1].get_color()
            plt.fill_between(np.arange(n), overall_fairness[i, :, 2] - fairness_std[i, :, 2],
                             overall_fairness[i, :, 2] + fairness_std[i, :, 2], alpha=0.4, color=color)

            plt.xlabel("Users")
            plt.ylabel("Group-Difference of \n Clicks per est. Relevance")
            ax= plt.gca()
            x0,x1 = ax.get_xlim()
            y0,y1 = ax.get_ylim()
        if only_two:
            plt.subplot(122)
        else:
            plt.subplot(144)
        #plt.title("Group-Difference \n Clicks per True Relevance")
        p = plt.plot(np.arange(n), overall_fairness[i,:,3], label=methods[i%len(methods)])# +" " + click_models[i//len(methods)])
        color = p[-1].get_color()
        plt.fill_between(np.arange(n), overall_fairness[i, :, 3] - fairness_std[i, :, 3],
                         overall_fairness[i, :, 3] + fairness_std[i, :, 3], alpha=0.4, color=color)

        #plt.axis([0,n,0,1])
        ax= plt.gca()
        ax.set_ylim(0, np.max(overall_fairness[i,15:,3]))
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))
        plt.xlabel("Users")
        plt.ylabel("Group-Difference of \n Clicks per True Relevance")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(PLOT_PREFIX + "Unfairness.pdf", bbox_inches="tight", dpi=800)
    #files.download("Unfairness.pdf")

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
        if "Naive" in model:
            fairness["rel"] = fairness["clicks"]
        
        #plt.plot(np.arange(n), fairness["rel"][:,g] / normalize_iter /len(G[g]) , label="Estimated Relevance")
        #plt.plot(np.arange(n), fairness["true_rel"][:,g] / normalize_iter /len(G[g]) , label="True Relevance")
       
        
        #plt.plot(np.arange(n), fairness["clicks"][:,g] / normalize_iter , label="Clicks")
        #plt.plot(np.arange(n), fairness["prop"][:,g] / normalize_iter /len(G[g]) , label="Propensities")

        plt.plot(np.arange(n), fairness["prop"][:,g] / fairness["rel"][:,g], label="Exposure per est. Relevance")
        plt.plot(np.arange(n), fairness["prop"][:,g] / fairness["true_rel"][:,g], label="Exposure per True Relevance")
        
        plt.plot(np.arange(n), fairness["clicks"][:,g] / fairness["rel"][:,g], label="Clicks per est. Relevance")
        plt.plot(np.arange(n), fairness["clicks"][:,g] / fairness["true_rel"][:,g], label="Clicks per True Relevance")
        
        plt.xlabel("Users")
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
        #plt.axis([0,n,0,1])
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

    plt.savefig(PLOT_PREFIX + "Fairness"+ model+ ".pdf", bbox_inches="tight")
    plt.show()
    #files.download("Fairness"+ model+ ".pdf")


def plot_ndcg(ndcg, label="", plot=True, figure_name="NDCG", window_size=0, std = None):
    plt.figure(figure_name)


    if window_size > 0:
        #moving_average = [np.mean(ndcg[i:i+window_size]) for i in range(len(ndcg)-window_size)]
        #moving_average = np.convolve(ndcg, np.ones((window_size))/window_size, mode='valid')
        #moving_average = [np.mean(ndcg[:(i+1)]) for i in range(len(ndcg))]
        moving_average = np.cumsum(ndcg)/np.arange(1,len(ndcg)+1)
        p = plt.plot(np.arange(len(moving_average)), moving_average, label=label, linestyle='-')
        #plt.axis([0, len(moving_average), 0.75, 0.9]
        plt.axis([0, len(moving_average), 0.6, 1])
        print(np.shape(ndcg), np.shape(moving_average), np.sum(ndcg), np.sum(moving_average))

        plt.ylabel("Avg. Cumulative NDCG")
        if(std is not None):
            color = p[-1].get_color()
            #print(np.shape(std))
            std = np.std(np.cumsum(std, axis=1) / np.arange(1, len(ndcg) + 1), axis=0)
            #print(np.shape(std),np.shape(ndcg),np.shape(moving_average))
            plt.fill_between(np.arange(len(moving_average)), moving_average - std, moving_average + std, alpha=0.4, color=color)
    else:
        p =  plt.plot(np.arange(len(ndcg)),ndcg, label = label)
        #plt.axis([0, len(ndcg), 0.75, 0.9])
        plt.axis([0, len(ndcg), 0.9, 1])

        if (std is not None):
            color = p[-1].get_color()
            plt.fill_between(np.arange(len(ndcg)), ndcg - std, ndcg + std, alpha=0.4,
                             color=color)

        plt.ylabel("NDCG")
    plt.xlabel("Users")
    if(plot):
        plt.title("Average NDCG in model " + label)
        plt.show()


def plot_NDCG_Unfairness(ndcg,unfairness,ax, ax2=None, label="", unfairness_label = "Unfairness", synthetic=False):
    #fig, ax =plt.subplots()
    #plt.figure(figure_name)
    trials, n = np.shape(ndcg)

    cum_ndcg = np.cumsum(ndcg, axis=1)/ np.arange(1,n+1)
    std_ndcg = np.std(cum_ndcg, axis=0)
    cum_ndcg = np.mean(cum_ndcg, axis=0)

    unfairness_std = np.std(unfairness,axis=0)
    unfairness = np.mean(unfairness, axis=0)
    p = ax.plot(np.arange(n), cum_ndcg, label=label, linestyle='-')

    color = p[-1].get_color()
    ax.set_xlim([0,n])
    if(synthetic):
        ax.set_ylim([0.95, 1])
    else:
        ax.set_ylim([0.75,0.9])
    if ax2 is not None:
        ax2.set_xlim([0,n])
        if(synthetic):
            ax2.set_ylim([0, 0.2])
        else:
            ax2.set_ylim([0, 0.2])
        ax2.set_xlabel("Users")
        ax2.set_ylabel(unfairness_label)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    #ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    ax.fill_between(np.arange(n), cum_ndcg - std_ndcg, cum_ndcg + std_ndcg, alpha=0.4, color=color)
    ax.set_xlabel("Users")
    ax.set_ylabel("Avg. Cumulative NDCG")

    if ax2 is not  None:
        #ax2 = ax.twinx()
        x0, x1 = ax2.get_xlim()
        y0, y1 = ax2.get_ylim()
        #ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        #ax2.set_aspect("equal")
        ax2.plot(np.arange(n),unfairness, label=label, color=color) #, linestyle=':',
        ax2.fill_between(np.arange(n), unfairness - unfairness_std, unfairness + unfairness_std, alpha=0.4, color=color)
    return ax2


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


def load_and_plot_all():
    global DATA_SET
    DATA_SET = True
    skyline_data = np.load("plots/Jokes/Skyline/Fairness_Data.npy") #Skyline
    exposure_data = np.load("plots/Jokes/Exposure/Fairness_Data.npy") #Naive Pop, IPS, Fair-E-IPS
    exposure_pers_data = np.load("plots/Jokes/ExposurePers/Fairness_Data.npy") # Naive Pop, Pers, Fair-E-Pers
    impact_data = np.load("plots/Jokes/ImpactPers/Fairness_Data.npy") # Fair-I-IPS, Fair-I-Pers

    synthetic_data = np.load("plots/SynteticOverview/Fairness_Data.npy")

    items = [ Joke(i) for i in np.arange(0,90)]
    G = assign_groups(items)
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]

    #NDCG plot with Naive, IPS, Pers, Skyline

    ndcg = [  np.mean(exposure_data[0]["NDCG"],axis=0), np.mean(exposure_data[1]["NDCG"],axis=0), np.mean(exposure_pers_data[1]["NDCG"],axis=0),  np.mean(skyline_data[0]["NDCG"],axis=0)]
    ndcg_std = [exposure_data[0]["NDCG"], exposure_data[1]["NDCG"], exposure_pers_data[1]["NDCG"], skyline_data[0]["NDCG"]]
    labels = ["Naive","IPS","Pers","Skyline-Pers"]
    #plt.figure("NDCG", figsize=(15,5))
    for i in range(4):
        plot_ndcg(ndcg[i],labels[i],plot=False,window_size=1,std=ndcg_std[i])
    plt.legend(ncol=2)
    plt.savefig("plots/Paper_Plots/SkylineNDCG.pdf", bbox_inches="tight", dpi=800)
    plt.show()
    plt.close("all")

    #Disparity in Exposure 1. Naive, IPS, Fair-E-IPS, 2. IPS, Pers, Fair-E-Pers

    run_data = [exposure_data[0], exposure_data[1], exposure_data[2], exposure_pers_data[1], exposure_pers_data[2]]
    ndcg_full = [x["NDCG"][:,:7000] for x in run_data]

    labels=["Naive","IPS","Fair-E-IPS","Pers","Fair-E-Pers"]

    overall_fairness = get_unfairness_from_rundata(run_data,pair_group_combinations,10,7000)
    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0, 1,3, 2,4],
                                     "plots/Paper_Plots/NDCG_UnfairExposure.pdf", 0)


    #combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0,1,2],"plots/Paper_Plots/NDCG_UnfairExposurePop.pdf", 0)
    #combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [1,3,4],
    #                                 "plots/Paper_Plots/NDCG_UnfairExposurePers.pdf", 0)



    # Disparity in Impact 1. Naive, IPS, Fair-I-IPS, 2. IPS, Pers, Fair-I-Pers


    run_data = [exposure_data[0], exposure_data[1], impact_data[0], exposure_pers_data[1], impact_data[1]]
    ndcg_full = [x["NDCG"][:,:7000] for x in run_data]
    labels = ["Naive", "IPS", "Fair-I-IPS", "Pers", "Fair-I-Pers"]
    overall_fairness = get_unfairness_from_rundata(run_data, pair_group_combinations, 10, 7000)

    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0,1, 3, 2, 4],"plots/Paper_Plots/NDCG_UnfairImpact.pdf", 1)

    #combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0,1,2],"plots/Paper_Plots/NDCG_UnfairImpactPop.pdf", 1)
    #combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [1,3,4],
    #                                 "plots/Paper_Plots/NDCG_UnfairImpactPers.pdf", 1)


    #Unfairness In Impact & Exposure for IPS, Fair-E-Pers, Fair-I-Pers
    run_data = [exposure_data[1],  exposure_pers_data[2], impact_data[1]]
    labels = ["IPS", "Fair-E-Pers", "Fair-I-Pers"]
    overall_fairness = get_unfairness_from_rundata(run_data, pair_group_combinations, 10, 7000)
    plot_Exposure_and_Impact_Unfairness(overall_fairness,labels, "plots/Paper_Plots/ImpactVSExposure.pdf")


    #Syntethic Data

    DATA_SET = False
    items = [Item(i) for i in np.linspace(-1, 1, 20)] + [Item(i) for i in np.linspace(-1, 0.01, 10)]
    G = assign_groups(items)
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]
    run_data = synthetic_data
    ndcg_full = [x["NDCG"][:, :5000] for x in run_data]
    labels = ["Naive", "IPS",  "Fair-I-IPS", "Fair-E-IPS"]
    overall_fairness = get_unfairness_from_rundata(run_data, pair_group_combinations, 100, 5000)

    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0, 1, 2],
                                     "plots/Paper_Plots/NDCG_UnfairImpactSynthetic.pdf", 1, synthetic=True)
    combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, [0, 1, 2],
                                     "plots/Paper_Plots/NDCG_UnfairExposureSynthetic.pdf", 0, synthetic=True)


def plot_Exposure_and_Impact_Unfairness(overall_fairness, labels, filename):
    n = np.shape(overall_fairness)[2]
    plt.figure("Unfairness", figsize=(15, 5))
    fairness_std = np.std(overall_fairness, axis=1)
    overall_fairness = np.mean(overall_fairness, axis=1)
    for i in range(len(labels)):
        plt.subplot(121)
        p = plt.plot(np.arange(n), overall_fairness[i, :, 0], label=labels[i])
        color = p[-1].get_color()
        plt.fill_between(np.arange(n), overall_fairness[i, :, 0] - fairness_std[i, :, 0],
                         overall_fairness[i, :, 0] + fairness_std[i, :, 0], alpha=0.4, color=color)
        ax = plt.gca()
        ax.set_ylim(0, 0.2)
        ax.set_xlim(0, n)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        #ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        plt.xlabel("Users")
        plt.ylabel("Exposure Unfairness")
        ax.legend()

        plt.subplot(122)
        p = plt.plot(np.arange(n), overall_fairness[i, :, 1], label=labels[i])
        color = p[-1].get_color()
        plt.fill_between(np.arange(n), overall_fairness[i, :, 1] - fairness_std[i, :, 1],
                         overall_fairness[i, :, 1] + fairness_std[i, :, 1], alpha=0.4, color=color)
        ax = plt.gca()
        ax.set_ylim(0, 0.2)
        ax.set_xlim(0, n)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        #ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        plt.xlabel("Users")
        plt.ylabel("Impact Unfairness")
        ax.legend()
    #plt.legend(loc=0)# 'upper right')
    plt.savefig(filename, bbox_inches="tight", dpi=800)

    plt.close("all")


def combine_and_plot_ndcg_unfairness(ndcg_full, overall_fairness, labels, selection, name,  type=0 ,synthetic=False):

    #fig, ax = plt.subplots()
    #ax2 = None
    ax = plt.subplot(121)
    ax2 = plt.subplot(122)
    unfairness_label = "Impact Unfairness" if type == 1 else "Exposure Unfairness"
    for i in selection:
        #ax2 = plot_NDCG_Unfairness(ndcg_full[i], overall_fairness[i, :, :, type], ax=ax, ax2=ax2, label=labels[i],
        plot_NDCG_Unfairness(ndcg_full[i], overall_fairness[i, :, :, type], ax=ax, ax2=ax2, label=labels[i],
                                   unfairness_label=unfairness_label,synthetic=synthetic)

    ax.legend(ncol=2)#title="NDCG",loc="upper left")
    ax2.legend(ncol=2)#title="Unfairness", loc ="upper right")
    #ax.legend(title="NDCG", loc="center left")
    #ax2.legend(title="Unfairness", loc="center right")

    #plt.legend(ax.lines + ax2.lines, ncol=2)
    #ax.legend(ncol=2)
    #squarify(fig)
    plt.savefig(name, bbox_inches="tight", dpi=800)
    plt.close("all")


def plot_with_errorbar(x, all_stats, methods, filename, x_label, log_x = False, impact=True):
    x = np.asarray(x)

    plt.figure("Errorbar", figsize=(15, 5))
    """
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    """
    ax = plt.subplot(121)
    ax2 = plt.subplot(122)

    if log_x:
        #ax.set_xscale('log')
        ax.set_xscale("symlog", linthreshx= x[0] if x[0] != 0 else x[1])
        ax.set_xticks(x)
        ax2.set_xscale("symlog", linthreshx= x[0] if x[0] != 0 else x[1])
        ax2.set_xticks(x)

    ax.set_xlabel(x_label)
    ax2.set_xlabel(x_label)
    ax.set_ylabel("NDCG")
    if impact:
        ax2.set_ylabel("Impact Unfairness")
        #ax.set_ylim([0.95, 1])
        ax.set_ylim([0.7, 1.05])
        ax2.set_ylim([0, 0.8])
    else:
        ax2.set_ylabel("Exposure Unfairness")
        ax2.set_ylim([0, 0.8])
        #ax2.set_ylim([0, 0.2])
        ax.set_ylim([0.7, 1.05])
        #ax.set_ylim([0.95, 1])
    ax.set_xlim(min(x),max(x))
    ax2.set_xlim(min(x), max(x))
    #ax.set_ylim([0.7, 1])

    for i, m in enumerate(methods):
        ax.errorbar(x *(1. + ((i*2 -1)/100.)), np.mean(all_stats[:, i, :, 0], axis=1), np.std(all_stats[:, i, :, 0], axis=1), label=m)
    for i, m in enumerate(methods):
        if impact:
            ax2.errorbar(x *(1. + ((i*2 -1)/100.)), np.mean(all_stats[:, i, :, 2], axis=1), np.std(all_stats[:, i, :, 2], axis=1), label=m)#, linestyle=":")
        else:
            ax2.errorbar(x *(1. + ((i*2 -1)/100.)), np.mean(all_stats[:, i, :, 1], axis=1), np.std(all_stats[:, i, :, 1], axis=1), label=m)#, linestyle=":")

    ax.legend( loc=0)# title="NDCG",loc="center left")
    ax2.legend( loc=0)#title="Unfairness", loc="center right")
    plt.savefig(filename, bbox_inches="tight", dpi=800)
    plt.close("all")


def load_and_plot_lambda_comparison():

    items = [Item(i) for i in np.linspace(-1, 1, 20)] +[Item(i) for i in np.linspace(-1, 0.01, 10)]
    G = assign_groups(items)
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]

    trials, iterations = 10, 2000
    methods = ["Fair-I-IPS-LP", "Fair-I-IPS"]
    n = len(methods)
    data = np.load("plots/LambdaTest/Fairness_Data.npy") # Fair-I-IPS-LP [0,2,4,6,8]  Fair-I-IPS [1,3,5,7,9] lambda 0.0001, 0.001, 0.01, 0.1, 10
    data100 = np.load("plots/LambdaTest/Fairness_Data100.npy")  # Fair-I-IPS-LP [0]  Fair-I-IPS [1]   lambda 100
    data1 = np.load("plots/LambdaTest/Fairness_Data1.npy")  # Fair-I-IPS-LP [0]  Fair-I-IPS [1]   lambda 1
    data1000 = np.load("plots/LambdaTest/Fairness_Data1000.npy")  # Fair-I-IPS-LP [0]  Fair-I-IPS [1]   lambda 1000, 10000
    """
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
    """

    data = np.load("plots/LambdaTest/Fairness_Data_Final.npy")

    trials, iterations = 10, 2000
    x = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    plot_lambda_comparison(x, data, methods, trials, pair_group_combinations, "lambdaImpactAllComparisons_final")


def plot_lambda_comparison(x,data, methods, trials, pair_group_combinations,name ):

    n = len(methods)
    all_stats = np.zeros((len(data) // 2, len(methods), trials, 3))
    for j in range(len(data)):
        all_stats[j // n, j % n, :, 0] = data[j]["NDCG"][:, -1]

        for a, b in pair_group_combinations:
            all_stats[j // n, j % n, :, 1] += np.abs(data[j]["prop"][:, -1, a] / data[j]["true_rel"][:, -1, a] -
                                                     data[j]["prop"][:, -1, b] / data[j]["true_rel"][:, -1, b])

            all_stats[j // n, j % n, :, 2] += np.abs(data[j]["clicks"][:, -1, a] / data[j]["true_rel"][:, -1, a] -
                                                     data[j]["clicks"][:, -1, b] / data[j]["true_rel"][:, -1, b])
    all_stats[:, :, :, 1:] /= len(pair_group_combinations)
    print("allstats shape", np.shape(all_stats))
    plot_with_errorbar(x, all_stats, methods, "plots/Paper_Plots/"+ name+ ".pdf",
                       r'$\lambda $', log_x=True)
    plot_with_errorbar(x, all_stats, methods,
                       "plots/Paper_Plots/" + name+ "Exposure.pdf",
                       r'$\lambda $', log_x=True, impact=False)