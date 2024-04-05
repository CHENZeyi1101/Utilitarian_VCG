import numpy as np;
import networkx as nx;
import matplotlib.pyplot as plt;
import random
import copy
import math
from copy import copy, deepcopy
from itertools import product
from network_class import *


def random_coloring(network, mycolor, seed = 42):
    np.random.seed(seed)
    rand_coloring = np.random.choice(mycolor, network.n).tolist()
    return rand_coloring

def color_sampling(color_set):
    return random.choice(color_set)

def utility_generator(network, n_color, seed = 42):
    utility = np.zeros((network.n, n_color))
    random.seed(seed)
    for i in range(network.n):
        for j in range(n_color):
            utility[i][j] = random.uniform(0, 1)
    return utility * 100

def activated_set(network, synchronic = False, omega = None):
    activated = []
    if not synchronic:
        activated.append(random.sample(range(network.n), 1)[0])
    else:
        random_list = [random.random() for _ in range(network.n)]
        for i in range(network.n):
            if random_list[i] <= omega:
                activated.append(i)
    return activated

def isconflict(network, coloring, i): # for a single vertex
    if coloring[i] in [coloring[j] for j in network.get_neighbors(i)]:
        return True
    else:
        return False
    
def isproper(network, coloring): # for the whole graph 
    for i in range(network.n):
        if isconflict(network, coloring, i):
            print("clash in ", i)
            return False
    return True

def welfare(network, coloring, utility, weight):
    welfare = 0
    for i in range(network.n):
        welfare += utility[i][coloring[i]] * weight[i] * (1 - isconflict(network, coloring, i))
    return welfare

def expected_loss(network, coloring, utility, weight):
    loss = 0
    for i in range(network.n):
        _, prob_inrisk = network.get_inrisk(i, coloring)
        # print(prob_inrisk)
        loss += (weight[i] * utility[i][coloring[i]] * (1 - isconflict(network, coloring, i))
                * (1 - np.prod(np.array([1]*len(prob_inrisk) - np.array(prob_inrisk)))))
    return loss

def weight_generator(elements, seed = 42):
    n = len(elements)
    random.seed(seed)
    weights = [random.random() for _ in range(n)]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return normalized_weights

def Jain_index(network, coloring, utility, weight):
    numerator = welfare(network, coloring, utility, weight) ** 2

    welfare_1 = welfare(network, coloring, utility, weight)
    welfare_2 = welfare(network, coloring, utility, [1] * network.n)
    return welfare_1 / welfare_2

