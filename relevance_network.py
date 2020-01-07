
"""##Relevance Estimating Neural Network"""
import numpy as np
import tensorflow as tf
import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch import optim
import keras
from keras.layers import Dense
from keras import regularizers

class relevance_estimating_network:

    def __init__(self, input_dim = 2, output_dim = 1, hidden_units = 16, joke = False, supervised = False, news = False ):
        self.joke = joke
        self.news = news
        self.supervised = supervised
        self.model = keras.models.Sequential()
        if hidden_units == 0:# or joke:
            self.model.add(Dense(output_dim, input_shape=(input_dim,), activation='sigmoid'))
        else:
            self.model.add(Dense(hidden_units,input_shape=(input_dim,), activation = 'relu'))
            #self.model.add(Dense(hidden_units, input_shape=(hidden_units,), activation='relu'))
            self.model.add(Dense(output_dim, activation = 'sigmoid'))
            #self.model.add(Dense(output_dim, activation='relu'))

        if (joke or news) and not supervised:
            self.model.compile(optimizer="adam", loss=self.unbiased_loss, metrics=[self.unbiased_loss])
        else:
            self.model.compile(optimizer="adam",loss="mse", metrics=['mse'])

    def unbiased_loss(self, y_true, y_pred):
        #click/propensity as objective
        return keras.backend.mean((0 - y_pred) **2 +  y_true * ((1-y_pred)**2 - (0-y_pred)**2))


    def train(self,features, relevances, x_test=None,y_test=None, epochs = 5000):
        self.model.fit(features, relevances, batch_size = len(features), verbose=0, epochs = epochs)
        train_score = self.model.evaluate(features,relevances,batch_size = len(features),verbose=0)

        #print("Training performance:" , train_score, "with {} items and {} avg. relevance".format(len(features),sum(relevances)/len(relevances)))
        #print(np.sort(relevances)[:5])
        if(x_test is not None):
            score = self.model.evaluate(x_test,y_test,batch_size = len(x_test))
            print("Evaluating performance:", score)
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test,y_test,batch_size = len(x_test))

    def predict(self, features):
        if self.joke or self.news:
            if len(np.shape(features))==1:
                result = self.model.predict(features[np.newaxis,:]).flatten()
            else:
                result = self.model.predict(features).flatten()
        else:
            result = self.model.predict(features).flatten()
        #print("predicted :", np.round(result,3), "containing {} elements ".format(len(result)))
        return result
    
class linear_one_hot_network(tnn.Module):
    def __init__(self, input_features = 2, input_items = 27, output_dim = 1):
        super(linear_one_hot_network, self).__init__()
       
        self.n = input_items        
        self.out_dim = output_dim
        self.n_features = input_features
        #self.weight = tnn.Parameter(torch.randn(output_dim, self.n+self.n_features))
        self.weight = tnn.Parameter(0.001 * torch.ones(output_dim, self.n+self.n_features))
        
    
    def forward(self, x):
        x = F.linear(x, self.weight)
        return x    

    def predict(self,x, original_items = True ):
        X = self.features_to_input(x, original_items )
        return self.forward(X).data.numpy().flatten()
    
    def add_item(self,n=1):
        #Add a new variable initialized close to zero for each new item
        self.n += n
        with torch.no_grad():
            self.weight = tnn.Parameter(torch.cat((self.weight, 0.001 * torch.ones(self.out_dim, n)), 1))
    
    def train(self, features, relevances, epochs = 5000, lr=0.01, reg_w=0.001, consider=None):            
        #Train on features and one-hot-encoding
        X = self.features_to_input(features, original_items = True)
        Y = torch.FloatTensor(relevances)
        if consider is not None:
            X = X[consider]
            Y = Y[consider]
        
        optimizer = optim.Adam(self.parameters(), lr=lr) 
                
        for epoch in range(epochs):
            optimizer.zero_grad() 
            loss = torch.norm(Y-torch.t(self.forward(X)))**2  + reg_w * torch.norm(self.weight[:,:self.n_features])**2 + 10*reg_w * torch.norm(self.weight[:,self.n_features:])**2 
            #if(epoch % 50 == 0):
                #print("Loss in epoch{} : ".format(epoch), loss)
                #print("Predicted relevances:", self.forward(X).data.numpy().flatten())
            loss.backward()
            optimizer.step() 
         
    def features_to_input(self, items, original_items = True):
        assert(np.shape(items)[-1] == self.n_features)
                       
        x = np.zeros((len(items),self.n+self.n_features))
        x[:,:self.n_features] = items

        if(original_items):
            #One-hot_encoding
            assert(len(items) == self.n)
            x[np.arange(self.n),self.n_features + np.arange(self.n)] = 1
        return torch.FloatTensor(x)
        
        
        