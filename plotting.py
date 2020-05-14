import numpy as np
import scipy.integrate
import scipy.stats
from matplotlib import pyplot as plt
import matplotlib as mpl
import data_utils
from Documents import Item, Movie
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


from Documents import Movie, Item


def assign_groups(items):
    n_groups = max([i.g for i in items])+1
    G = [ [] for i in range(n_groups)]
    for i, item in enumerate(items):
        G[item.g].append(i)
    return G


@ex.capture
def plot_neural_error(errors, labels, PLOT_PREFIX):
    plt.close("all")
    # Plot Neural error
    for i, error in enumerate(errors):
        plt.plot(np.arange(100, len(error)+100), error, label=labels[i])
    plt.legend()
    plt.ylabel("Difference True and estimated relevance")
    plt.xlabel("Users")
    plt.savefig(PLOT_PREFIX + "Neural_Error.pdf", bbox_inches="tight")

@ex.capture
def plot_click_bar_plot(frac_c,labels, save=False, PLOT_PREFIX=""):
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

@ex.capture
def plot_unfairness_over_time(overall_fairness, click_models, methods, only_two = True, PLOT_PREFIX=""):
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

@ex.capture
def plot_fairness_over_time(fairness, G, model, PLOT_PREFIX, DATA_SET):
    n = np.shape(fairness["rel"])[0]
    if DATA_SET == 0:
        group_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
        color_dict = {0: "blue", 1: "red", 2: "black"}
    else:
        group_dict = {0: "C0", 1: "C1", 2: "C2", 3: "C3", 4: "C4"}
        color_dict = {0: "blue", 1: "red", 2: "black", 3: "green", 4: "yellow"}
    normalize_iter = np.arange(n)
    normalize_iter[0] = 1#Zerost iteration should be divided by 1

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
    plt.savefig(name, bbox_inches="tight", dpi=800)
    plt.close("all")



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
    """
    if(synthetic):
        ax.set_ylim([0.95, 1])
    else:
        ax.set_ylim([0.75,0.9])
        """
    if ax2 is not None:
        ax2.set_xlim([0,n])
        """
        if(synthetic):
            ax2.set_ylim([0, 0.2])
        else:
            ax2.set_ylim([0, 0.2])
        """
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
        ax.set_ylim(0, 0.4)
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
        ax.set_ylim(0, 0.4)
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
        if np.ndim(overall_fairness)==4:
            plot_NDCG_Unfairness(ndcg_full[i], overall_fairness[i, :, :, type], ax=ax, ax2=ax2, label=labels[i],
                                       unfairness_label=unfairness_label,synthetic=synthetic)
        else:
            plot_NDCG_Unfairness(ndcg_full[i], overall_fairness[i, :, :], ax=ax, ax2=ax2, label=labels[i],
                                 unfairness_label=unfairness_label, synthetic=synthetic)
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


def load_and_plot_lambda_comparison(data_path, trials):

    items = [Item(i) for i in np.linspace(-1, 1, 20)] +[Item(i) for i in np.linspace(-1, 0.01, 10)]
    G = assign_groups(items)
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]
    methods = ["Fair-I-IPS-LP", "Fair-I-IPS"]
    data = np.load(data_path + "Fairness_Data.npy")

    x = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    plot_lambda_comparison(x, data, methods, trials, pair_group_combinations, data_path + "lambdaAllComparisons")


def plot_lambda_comparison(x, data, methods, trials, pair_group_combinations, name):

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
    #print("allstats shape", np.shape(all_stats))
    plot_with_errorbar(x, all_stats, methods, name+ "Impact.pdf",
                       r'$\lambda $', log_x=True)
    plot_with_errorbar(x, all_stats, methods, name+ "Exposure.pdf",
                       r'$\lambda $', log_x=True, impact=False)




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
