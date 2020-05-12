"""## Item Class"""
import numpy as np

#GROUP_BOUNDARIES = [[-1,-0.33],[0.33,1], [-0.33,0.33]] #Boundaries for Left, Right, Neutral
GROUP_BOUNDARIES = [[-1,-0],[0,1]] #Boundaries for Left and Right


class Item:
    def __init__(self, polarity, quality=1, news_group = None, id=0):
        """
        Creates an Document/article
        Assigns a Group depending on polarity

        """
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


class Movie:

    def __init__(self, id, group):
        self.id = id
        self.g = group
