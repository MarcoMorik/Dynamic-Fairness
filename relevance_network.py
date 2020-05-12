
"""##Relevance Estimating Neural Network"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
from keras import regularizers


class relevance_estimating_network:

    def __init__(self, input_dim = 2, output_dim = 1, hidden_units = 16, supervised = False, news = False, logdir="data/"):

        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        self.history =  []
        self.news = news
        self.supervised = supervised
        self.model = keras.models.Sequential()
        if hidden_units == 0:# or joke:
            self.model.add(Dense(output_dim, input_shape=(input_dim,), activation='relu', kernel_regularizer=regularizers.l2(0.00001)))

        else:
            self.model.add(Dense(hidden_units,input_shape=(input_dim,), activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))
            self.model.add(Dense(output_dim, activation = 'sigmoid'))

        if news and not supervised:
            self.model.compile(optimizer="adam", loss=self.unbiased_loss, metrics=[self.unbiased_loss])
        else:
            self.model.compile(optimizer="adam",loss="mse", metrics=['mse'])


    def unbiased_loss(self, y_true, y_pred):
        #click/propensity as objective
        return keras.backend.mean((0 - y_pred) **2 +  y_true * ((1-y_pred)**2 - (0-y_pred)**2))


    def train(self,features, relevances, x_test=None,y_test=None, epochs = 50, trial=0):

        history = self.model.fit(features, relevances, batch_size = 100, verbose=0, epochs=epochs)# , callbacks=[self.tensorboard_callback])

        if(trial%1000 ==99):
            train_score = self.model.evaluate(features,relevances, batch_size = len(features), verbose=0)
            print("trial{}, loss:".format(trial), train_score)

        if(x_test is not None):
            score = self.model.evaluate(x_test,y_test,batch_size = len(x_test))
            print("Evaluating performance:", score)
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test,y_test,batch_size = len(x_test))

    def predict(self, features):
        if self.news:
            if len(np.shape(features))==1:
                result = self.model.predict(features[np.newaxis,:]).flatten()
            else:
                result = self.model.predict(features).flatten()
        else:
            result = self.model.predict(features).flatten()
        #print("predicted :", np.round(result,3), "containing {} elements ".format(len(result)))
        return result

