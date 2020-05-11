"""
Controlling Fairness and Bias in Dynamic Learning-to-Rank
#TODO Readme
author: Marco Morik
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config import ex
import numpy as np
from scipy.stats import truncnorm
import scipy.integrate
import scipy.stats
import random
#from sinkhorn_knopp import sinkhorn_knopp as skp
import pandas as pd
import time
import warnings; warnings.simplefilter('ignore') ##Ignores Warnings for nicer Plots. Disable for Debugging
import data_utils
import os
import birkhoff
import relevance_network
from itertools import permutations

from Documents import Item, Movie, Joke
import plotting

plotting.init_plotting()

birkhoff.TOLERANCE = 10**(-8)


