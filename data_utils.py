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
