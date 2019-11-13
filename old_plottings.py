def debug_beta():
    global sample_user
    global get_numerical_relevances
    sample_user = lambda: sample_user_base(beta=0.1)
    get_numerical_relevances = lambda x: get_numerical_relevances_base(x, beta=0.1)
    PLOT_PREFIX = "plots/Test_Beta/"
    items = [Item(i) for i in np.linspace(-1, 1, 20)]
    popularity = np.ones(len(items))
    methods = ["IPS", "Fair-I-IPS", "Fair-E-IPS"]
    trials = 1
    iterations = 20000

    collect_relevance_convergence(items, popularity, trials, click_models=["PBM_log"],
                                 methods=methods, iterations=iterations,
                                 plot_individual_fairness=True)
