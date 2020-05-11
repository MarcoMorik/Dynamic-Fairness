JOKE_THRESHHOLD = 2
import pandas as pd
import numpy as np

import os
from config import ex
#from scipy.sparse.linalg import svd

import surprise
sigmoid = lambda x: 1. / (1. + np.exp(-(x - 3) / 0.1))


def load_data():
    df = pd.read_excel("data/jester-data-1.xls",header=None)
    df2 = pd.read_excel("data/jester-data-2.xls",header=None)

    df = df.append(df2, ignore_index = True)

    complete = df[0] == 100

    df = df[complete].sample(frac=1)
    df = df.iloc[:, 1:]

    df = df.where(df > JOKE_THRESHHOLD, 0)
    df = df.where(df < JOKE_THRESHHOLD, 1)
    #df = df/10.
    feature_jokes = [5,7,8,13,15,16,17,18,19,20]
    no_features = [i for i in range(df.shape[1]) if i + 1 not in feature_jokes]
    joke = df.iloc[:, no_features]
    feature = df.iloc[:, feature_jokes]

    return joke, feature

def load_news_data(seed=18):
    #MEDIA_SOURCES = ["ABC","AP", "BBC", "Bloomberg", "Breitbart","Buzzfeed","CBS","CNN","Conservative Tribune", "Daily Mail", "Democrazy Now", "Fox News", "Huffington Post", "Intercept", "Life News", "MSNBC", "National Review", "New York Times", "The American Conservative", "The Federalist", "The Guardian", "Washington Post", "WorldNetDaily"]
    MEDIA_SOURCES = ["Breitbart", "CNN",
                     "Daily Mail", "Fox News", "Huffington Post", "MSNBC",
                     "New York Times", "The American Conservative", "The Guardian", "WorldNetDaily"]
    df = pd.read_csv("data/InteractiveMediaBiasChart.csv",)
    df["Group"] = df.Source.astype("category").cat.codes
    df["Bias"] /= 40
    df["Quality"] /= 62

    selector = df["Source"].isin(MEDIA_SOURCES)
    df_small = df[selector]
    #TODO shorten data
    df_tiny = pd.DataFrame(columns=df_small.columns)
    #print("LA")
    for source in MEDIA_SOURCES:
        one_source = df["Source"] == source
        x = df[one_source].sample(3,random_state=seed)
        #print("Subsample:", x)
        df_tiny = df_tiny.append(x)
    df_tiny["Group"] = df_tiny.Source.astype("category").cat.codes
    df["Group"] = df.Source.astype("category").cat.codes

    return df, df_small, df_tiny

def define_genre(meta_data):

    #Defining Genres
    #Generates a List of Lists of Movies in each Genre
    genres = []
    movie_g_id = []
    for ge in meta_data["genres"]:
        movie_g_id.append([])
        for temp in eval(ge):
            movie_g_id[-1].append(temp["id"])
            if temp not in genres:
                genres.append(temp)
    g_idx = [g["id"] for g in genres]
    #Modify the Genres according to the Group Id.
    meta_data["genres"] = meta_data["genres"].map(lambda xx: [xxx["id"] for xxx in eval(xx)])
    return meta_data, g_idx

def select_companies(meta_data):
    # Selecting Companies
    # MGM, Warner Bros, Paramount, 20th Fox, Columbia (x2)
    # 5 Movie Companies with most user ratings
    selected_companies = [1, 2, 3, 4, 7, 8]
    comp_to_group = [0, 1, 2, 3, 4, 4]

    comp = meta_data["production_companies"].value_counts().index[selected_companies]
    comp_dict = dict([(x, comp_to_group[i]) for i, x in enumerate(comp)])
    meta_data = meta_data.astype({"id": "int"})
    meta_data = meta_data[meta_data["production_companies"].isin(comp)]

    return meta_data, comp_dict

def select_movies(ratings, meta_data, n_movies = 100, n_user= 10000):

    # Use the 100 Movies with the most ratings
    po2 = ratings["movieId"].value_counts()

    #Select the n_movies with highest variance
    var_scores = [np.std(ratings[ratings["movieId"].isin([x])]["rating"]) for x in po2.index[:(n_movies*3)]]
    var_sort = np.argsort(var_scores)[::-1]

    selected_movies = po2.index[var_sort[:n_movies]]
    #selected_movies =po2.index[:n_movies]
    ratings = ratings[ratings["movieId"].isin(selected_movies)]
    meta_data = meta_data[meta_data["id"].isin(selected_movies)]

    po = ratings["userId"].value_counts()

    ratings = ratings[ratings["userId"].isin(po.index[:n_user])] # remove users with less than 10 votes
    meta_data = meta_data[meta_data["id"].isin(ratings["movieId"].value_counts().index[:])]

    return ratings, meta_data

def get_user_features_genre(ratings, ratings_full, meta_data, n_user, g_idx):
    # Generate User Features (Mean rating on each movie Genre)
    n_user = len(ratings["userId"].unique())
    user_features = np.zeros((n_user, len(g_idx)))
    user_id_to_idx = dict(zip(sorted(ratings["userId"].unique()), np.arange(n_user)))
    temp = pd.merge(ratings_full[ratings_full["userId"].isin(ratings["userId"].unique())], meta_data, left_on="movieId",
                    right_on="id")
    for j, g_id in enumerate(g_idx):
        temp2 = temp[[g_id in x for x in temp["genres"]]]
        ids = [user_id_to_idx[x] for x in temp2["userId"].unique()]
        user_features[ids, j] = temp2.groupby('userId')["rating"].mean()

    return user_features

def get_ranking_matrix_incomplete(ratings, meta_data, n_user):

    user_id_to_idx = dict(zip(sorted(ratings["userId"].unique()), np.arange(n_user)))
    # Create a single Ranking Matrix, only relevance for rated movies
    # Leave it incomplete
    # ranking_matrix = np.zeros((n_user, n_movies))
    ranking_matrix = np.zeros((n_user, len(meta_data["id"])))
    movie_id_to_idx = {}
    movie_idx_to_id = []
    print(np.shape(ranking_matrix))
    for i, movie in enumerate(y["id"]):
        movie_id_to_idx[movie] = i
        movie_idx_to_id.append(movie)
        single_movie_ratings = ratings[ratings["movieId"].isin([movie])]
        ranking_matrix[[user_id_to_idx[x] for x in single_movie_ratings["userId"]], i] = single_movie_ratings[
            "rating"]

    return ranking_matrix
def get_matrix_factorization(ratings, meta_data, n_user, n_movies):
    # Matrix Faktorization
    algo = surprise.SVD(n_factors=50, biased=False)
    reader = surprise.Reader(rating_scale=(0.5, 5))
    surprise_data = surprise.Dataset.load_from_df(ratings[["userId", "movieId", "rating"]],
                                                  reader).build_full_trainset()
    algo.fit(surprise_data)

    pred = algo.test(surprise_data.build_testset())
    print("MSE: ", surprise.accuracy.mse(pred))
    print("RMSE: ", surprise.accuracy.rmse(pred))

    ranking_matrix = np.dot(algo.pu, algo.qi.T)
    # ranking_matrix = np.clip(ranking_matrix, 0.5, 5)

    # movie_idx_to_id = [surprise_data.to_raw_iid(x) for x in movies_to_pick]
    movie_idx_to_id = [surprise_data.to_raw_iid(x) for x in range(n_movies)]
    features_matrix_factorization = algo.pu
    print("Means: ", np.mean(features_matrix_factorization), np.mean(algo.qi.T))
    print("Feature STD:", np.std(features_matrix_factorization), np.std(algo.qi))
    print("Full Matrix Shape", np.shape(ranking_matrix), "rankinG_shape", np.shape(ranking_matrix))

    return ranking_matrix, features_matrix_factorization, movie_idx_to_id

@ex.capture
def load_movie_data(n_movies, n_user, n_company, movie_features, movie_ranking_sample_file =None):
    """
    Loads the Movie Dataset
    Preprocesses
    Generate Rating Matrices with Matrix Factoriation
    Sample Rankings from those Matrices.
    """
    #Loading Meta Data from the Movies
    meta_data = pd.read_csv("data/movies_metadata.csv")[["production_companies", "id", "genres"]]
    #Delete Movies with Date as ID
    meta_data = meta_data.drop([19730, 29503, 35587]) # No int id

    #Get Genres
    meta_data, g_idx = define_genre(meta_data)

    #Filter by Production Company to obtain 5 Groups
    meta_data, comp_dict = select_companies(meta_data)

    #Y = meta data from selected Companies

    #Load Ratings
    ratings_full = pd.read_csv("data/ratings.csv")
    ratings = ratings_full[ratings_full["movieId"].isin(meta_data["id"])]
    ratings, meta_data = select_movies(ratings, meta_data, n_movies, n_user)

    #Complete Ranking Matrix
    ranking_matrix = get_ranking_matrix_incomplete(ratings, meta_data, n_user)
    full_matrix, features_matrix_factorization, movie_idx_to_id = get_matrix_factorization(ratings, meta_data, n_user, n_movies)
    #Add the real rating for already rated movies
    full_matrix[np.nonzero(ranking_matrix)] = ranking_matrix[np.nonzero(ranking_matrix)]

    if movie_features == "factorization":
        user_features = features_matrix_factorization
    else:
        user_features = get_user_features_genre(ratings,ratings_full, meta_data, n_user,g_idx)

    #Generate Probability Matrix
    #ranking_matrix = np.clip((full_matrix - 1) / 4, a_min=0, a_max=1)
    prob_matrix = sigmoid(full_matrix)

    groups = [comp_dict[meta_data[meta_data["id"].isin([x])]["production_companies"].to_list()[0]] for x in
              movie_idx_to_id]

    po = ratings["userId"].value_counts()
    po2 = ratings["movieId"].value_counts()
    print("Number of Users", len(po.index), "Number of Movies", len(po2.index))
    print("the Dataset before completion is", len(ratings) / float(n_user*n_movies), " filled")
    print("The most rated movie has {} votes, the least {} votes; mean {}".format(po2.max(), po2.min(), po2.mean()))
    print("The most rating user rated {} movies, the least {} movies; mean {}".format(po.max(), po.min(), po.mean()))

    #The list of groups contains all movies
    assert(np.shape(groups) == (n_movies,))
    if movie_ranking_sample_file:
        for i in range(10):
            random_matrix = np.random.rand(n_user, n_movies)
            np.save(movie_ranking_sample_file+"{}.npy".format(i), [np.asarray(prob_matrix > random_matrix, dtype=np.float16),user_features, groups])

    return prob_matrix, user_features, groups

def load_movie_data_saved(filename ="data/movie_data_prepared.npy"):
    """
    Load an already created Movie Rating Matrix
    """
    full_matrix, user_features, groups = np.load(filename, allow_pickle=True)
    return full_matrix, user_features, groups


def filter_to_small(ratings, id, n):
    x = ratings[id].value_counts()
    x = x[x>=n]
    return ratings[ratings[id].isin(x.index)]



#Looking for Group of Movies with at N Users rating all of them
def find_biclique(ratings, sol = [], depth = 0, M = 10, N = 500):
    """ Could not find a group with 10 Movies and N>=500
    Therefore Matrix Factorization"""
    print(sol)
    if len(sol) >= M:
        print("10 Movies found:", sol)
        return sol
    candidates = ratings["movieId"].value_counts().index[depth:]
    if len(candidates) == 0:
        return -1
    for c in candidates:
    #sol.append(candidates[depth])
        if c in sol:
            continue
        ratings2 = ratings[ratings["userId"].isin(ratings[ratings["movieId"].isin([c])]["userId"])]
        ratings2 = filter_to_small(ratings2, "movieId", N)
        ratings2 = filter_to_small(ratings2, "userId", M - len(sol))

        sol2 = find_biclique(ratings2, sol + [c], depth + 1, M, N)
        #print(sol2)
        if sol2 != -1:
            return sol2
        ratings = ratings[ratings["movieId"] != c]
    return -1


@ex.capture
def sample_user_base(distribution, alpha, beta, u_std, BI_LEFT = 0.5):
    """
    Returns a User of the News Platform
    A user cosists of is Polarity and his Openness
    """
    if(distribution == "beta"):
        u_polarity = np.random.beta(alpha, beta)
        u_polarity *= 2
        u_polarity -= 1
        openness = u_std
        #std = np.random.rand()*0.8 + 0.2
    elif(distribution == "discrete"):
        #3 Types of user -1,0,1. The neutral ones are more open
        u_polarity = np.random.choice([-1,0,1])
        if(u_polarity == 0):
            openness = 0.85
        else:
            openness = 0.1
    elif(distribution == "bimodal"):
        if np.random.rand() < BI_LEFT:
            #user = truncnorm.rvs(-1,1,0.5,0.2,1)
            u_polarity = np.clip(np.random.normal(0.5,0.2,1),-1,1)
        else:
            #user = truncnorm.rvs(-1,1,-0.5,0.2,1)
            u_polarity = np.clip(np.random.normal(-0.5, 0.2, 1), -1, 1)
        openness = np.random.rand()/2 + 0.05 #Openness uniform Distributed between 0.5 and 0.55
    else:
        print("please specify a distribution for the user")
        return (0,1)
    return np.asarray([u_polarity, openness]) #, user**2, np.sign(user)) #

def sample_user_joke():
    """
    Yielding a Joke
    #TODO returning a Joke more Consistent?
    """
    df, features = load_data()
    while True:
        for i in range(df.shape[0]):
            yield (df.iloc[i].as_matrix(), features.iloc[i].as_matrix())
        print("All user preferences already given, restarting with the old user!")
        new_ordering = np.random.permutation(df.shape[0])
        df = df.iloc[new_ordering]
        features = features.iloc[new_ordering]

@ex.capture
def sample_user_movie(MOVIE_RATING_FILE):
    """
    Yielding a Movie
    """
    ranking, features, _ = load_movie_data_saved(MOVIE_RATING_FILE)
    print(np.shape(ranking))
    while True:
        random_order = np.random.permutation(np.shape(ranking)[0])
        for i in random_order:
            yield (ranking[i,:], features[i,:])
        print("All user preferences already given, restarting with the old user!")







@ex.capture
def get_user_generator(DATA_SET):
    if DATA_SET == 1:
        sample_user_generator = sample_user_joke()
        sample_user = lambda: next(sample_user_generator)

    elif DATA_SET == 0:
        sample_user = lambda: sample_user_base(distribution="bimodal")

    elif DATA_SET == 2:
        sample_user_generator = sample_user_movie()
        sample_user = lambda: next(sample_user_generator)

    return sample_user