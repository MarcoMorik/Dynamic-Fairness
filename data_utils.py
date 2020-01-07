JOKE_THRESHHOLD = 2
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch import optim
import keras
from keras.layers import Dense
from keras import regularizers
#from scipy.sparse.linalg import svd

import surprise

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


def load_movie_data(n_movies=100, n_user=10000, n_company=5):
    x = pd.read_csv("data/movies_metadata.csv")[["production_companies", "id", "genres"]]
    x = x.drop([19730, 29503, 35587]) # No int id

    genres = []
    movie_g_id = []
    for ge in x["genres"]:
        movie_g_id.append([])
        for temp in eval(ge):
            movie_g_id[-1].append(temp["id"])
            if temp not in genres:
                genres.append(temp)
    g_idx = [g["id"] for g in genres]
    x["genres"] = x["genres"].map(lambda xx: [xxx["id"] for xxx in eval(xx)])

    comp = x["production_companies"].value_counts().index[1:n_company+1]
    comp_dict = dict([(x,i) for i,x in enumerate(comp)])
    x = x.astype({"id": "int"})
    y = x[x["production_companies"].isin(comp)]
    #y = y.astype({"id": "int"})

    ratings_full = pd.read_csv("data/ratings.csv")
    ratings = ratings_full[ratings_full["movieId"].isin(y["id"])]

    # Use the 100 Movies with the most ratings
    po2 = ratings["movieId"].value_counts()
    ratings = ratings[ratings["movieId"].isin(po2.index[:n_movies])]
    y = y[y["id"].isin(po2.index[:n_movies])]

    po = ratings["userId"].value_counts()
    ratings = ratings[ratings["userId"].isin(po.index[:n_user])] # remove users with less than 10 votes


    #Generate User Features (Mean rating on each movie Genre)
    user_features = np.zeros((n_user, len(g_idx)))
    user_id_to_idx = dict(zip(sorted(ratings["userId"].unique()), np.arange(n_user)))
    temp = pd.merge(ratings_full[ratings_full["userId"].isin(ratings["userId"].unique())], x, left_on="movieId", right_on="id")
    for j, g_id in enumerate(g_idx):
        temp2 = temp[[g_id in x for x in temp["genres"]]]
        ids = [user_id_to_idx[x] for x in temp2["userId"].unique()]
        user_features[ids, j] = temp2.groupby('userId')["rating"].mean()

    #Create a single Ranking Matrix, only relevance for rated movies

    #Leave it incomplete
    ranking_matrix = np.zeros((n_user, n_movies))
    movie_id_to_idx = {}
    movie_idx_to_id = []
    for i, movie in enumerate(y["id"]):
        movie_id_to_idx[movie] = i
        movie_idx_to_id.append(movie)
        single_movie_ratings = ratings[ratings["movieId"].isin([movie])]
        ranking_matrix[[user_id_to_idx[x] for x in single_movie_ratings["userId"]], i] = single_movie_ratings["rating"]
    #Group(movie) = Company(movie)
    groups = [comp_dict[y[y["id"].isin([x])]["production_companies"].to_list()[0]] for x in movie_idx_to_id ]

    #Matrix Faktorization
    """engine = mf.MF(nr_iters=50, k= 20, quiet=True)
    engine.fit([[x, y, ranking_matrix[x, y]] for x, y in zip(*ranking_matrix.nonzero())])
    full_matrix = engine.predict(np.asarray([[i,j] for i in range(n_user) for j in range(n_movies)]))
    """

    algo = surprise.SVD(biased=False)
    reader = surprise.Reader(rating_scale=(0.5, 5))
    surprise_data = surprise.Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader).build_full_trainset()
    algo.fit(surprise_data)

    full_matrix = np.dot(algo.pu, algo.qi.T)
    #full_matrix = np.clip(full_matrix, 0.5, 5)
    movie_idx_to_id = [surprise_data.to_raw_iid(x) for x in range(n_movies)]
    groups = [comp_dict[y[y["id"].isin([x])]["production_companies"].to_list()[0]] for x in movie_idx_to_id ]

    features_matrix_factorization = algo.pu


    feature_movie_watched = np.zeros((n_user,n_movies))
    feature_movie_watched[np.nonzero(ranking_matrix)] = 1

    po = ratings["userId"].value_counts()
    po2 = ratings["movieId"].value_counts()
    print("the Dataset before completion is", len(ratings) / float(n_user*n_movies), " filled")
    print("The most rated movie has {} votes, the least {} votes; mean {}".format(po2.max(), po2.min(), po2.mean()))
    print("The most rating user rated {} movies, the least {} movies; mean {}".format(po.max(), po.min(), po.mean()))

    assert(np.shape(ranking_matrix)==(n_user,n_movies))
    assert(np.shape(groups) == (n_movies,))
    assert(np.shape(user_features)[0] == n_user)


    #Transform matrix to click probability
    #ranking_matrix = np.clip( (ranking_matrix-1) / 4, a_min=0, a_max=1)
    ranking_matrix = np.clip((full_matrix - 1) / 4, a_min=0, a_max=1)

    #user_features = features_matrix_factorization
    user_features /= 5


    user_features = np.concatenate((user_features,feature_movie_watched),axis=1)
    print(np.shape(user_features))
    np.save("data/movie_data_full_120features.npy", [ranking_matrix, user_features, groups])
    return ranking_matrix, user_features, groups

def load_movie_data_saved(filename ="data/movie_data_prepared.npy"):
    full_matrix, user_features, groups = np.load(filename)
    return full_matrix, user_features, groups


def filter_to_small(ratings, id, n):
    x = ratings[id].value_counts()
    x = x[x>=n]
    return ratings[ratings[id].isin(x.index)]


def find_biclique(ratings, sol = [], depth = 0, M = 10, N = 500):
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

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    """Matrix Factorization:
    Inputs:
    R: Ratings     (NxM)
    P: Random Faktor (NXK)
    Q: Random Faktor (MxK)
    K:  Numver of latent factors

    """
    Q = Q.T
    not_zero = R.nonzero()
    for step in range(steps):


        eR = R - np.dot(P, Q)
        """
        for k in range(K):
            P[not_zero[0], k] = P[not_zero[0], k] + alpha * (2 * eR[not_zero] * Q[k, not_zero[1]] - beta * P[not_zero[0], k])
            Q[k, not_zero[1]] = Q[k, not_zero[1]] + alpha * (2 * eR[not_zero] * P[not_zero[0], k] - beta * Q[k, not_zero[1]])
        """

        for i, j in zip(*not_zero):
            #P[i,:] = P[i,:] + alpha * (2 * eR[i,j][np.newaxis] * Q[:,j] - beta * P[i,:])
            #Q[:,j] = Q[:,j] + alpha * (2 * eR[i,j] * P[i,:] - beta * Q[:,j])
            for k in range(K):
                P[i][k] = P[i][k] + alpha * (2 * eR[i,j]* Q[k][j] - beta * P[i][k])
                Q[k][j] = Q[k][j] + alpha * (2 * eR[i,j] * P[i][k] - beta * Q[k][j])

        if np.mean(eR[not_zero]**2) < 0.001:
            break
        if not (step % 10):
            print(step, "error : ", np.mean(eR[not_zero]**2))
    return P, Q.T

def matrix_factorization_2(R, K):
    not_zero = R.nonzero()
    u_means = np.mean(R, axis=1)
    U, sigma, Vt = svds(R- u_means[:,np.newaxis],k=K)
    pred = np.dot(np.dot(U, sigma), Vt) + u_means[:,np.newaxis]

class relevance_estimating_network:

    def __init__(self, input_dim = 2, output_dim = 1, hidden_units = 16):
        self.model = keras.models.Sequential()
        #self.model.add(Dense(hidden_units,input_shape=(input_dim,), activation = 'relu'))
        #self.model.add(Dense(output_dim, activation = 'sigmoid'))
        self.model.add(Dense(output_dim, input_shape=(input_dim,), activation='sigmoid'))
        self.model.compile(optimizer="adam",loss="mse", metrics=['mse'])

    def train(self,features, relevances, x_test=None,y_test=None, epochs = 5000):
        self.model.fit(features,relevances, batch_size = min([len(features),400]), verbose=1, epochs = epochs)
        train_score = self.model.evaluate(features,relevances,batch_size = len(features),verbose=1)

        print("Training performance:" , train_score, "with {} items and {} avg. relevance".format(len(features),sum(relevances)/len(relevances)))
        #print(np.sort(relevances)[:5])
        if(x_test is not None):
            score = self.model.evaluate(x_test,y_test,batch_size = len(x_test))
            print("Evaluating performance:", score)
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test,y_test,batch_size = len(x_test))

    def predict(self,features):
        result = self.model.predict(features).flatten()
        #print("predicted :", np.round(result,3), "containing {} elements ".format(len(result)))
        return result


def test_neural_networks_jokes():
    jokes, features = load_data()
    no_features = [i for i in range(jokes.shape[1]) if i+1 not in features]
    joke = jokes.iloc[:, no_features]
    feature = jokes.iloc[:, features]

    print(joke.shape, feature.shape, jokes.shape)
    x = jokes.groupby(features).mean()
    x = x.where(x < 0.5, lambda y: 1 - y)

    train_x = feature.iloc[:10000]
    train_x = np.ones(10000)
    train_y = joke.iloc[:10000]
    test_x = feature.iloc[10000:]

    test_y = joke.iloc[10000:]
    test_x = np.ones(test_y.shape[0])
    nn = relevance_estimating_network(1,100,10)
    nn.train(train_x,train_y, test_x, test_y, epochs = 500)





#test_neural_networks_jokes()
