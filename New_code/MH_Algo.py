import numpy as np;
import networkx as nx;
import matplotlib.pyplot as plt;
import random
import copy
import math
from copy import copy, deepcopy
from itertools import product
from coloring_functions import *
from network_class import *

def delta_local_utility(network, i, neighbors, utility_matrix, c_old, c_prime, weight): # delta_U
    neighbor_before = 0
    neighbor_after = 0
    for j in neighbors:
        neighbor_before += utility_matrix[j][c_old[j]] * weight[j] * (1 - isconflict(network, c_old, j))
        neighbor_after += utility_matrix[j][c_prime[j]] * weight[j] * (1 - isconflict(network, c_prime, j))
    difference = (neighbor_after 
                - neighbor_before 
                + utility_matrix[i][c_prime[i]] * weight[i] * (1 - isconflict(network, c_prime, i))
                - utility_matrix[i][c_old[i]] * weight[i] * (1 - isconflict(network, c_old, i)))
    return difference

def MH_policy_async(network, mycolor, weight, utility, T =2000, temp_type = 0):
    c = random_coloring(network, mycolor, seed = 20) # initialization
    # print(c)
    utility_matrix = utility
    welfare_record = [welfare(network, c, utility_matrix, weight)]
    for t in range(T):
        if temp_type == 0: # trigonometric additive cooling
            temp = 0.01 + 0.5 * (10 - 0.01)*(1 + np.cos(np.pi * t / T))
        if temp_type == 1: # multiplicative cooling
            temp = 10 * np.power(0.99, t)
        elif temp_type == 2: # logarithmical cooling
            temp = 1 / (1 + np.log(1 + 100 * t))
        elif temp_type == 3: # constant
            temp = 0.01
        c_old = deepcopy(c)
        # print("old color: ", c_old)
        activated = activated_set(network, synchronic=False)
        # print("activated agents: ", activated)
        for i in activated:
            c_prime = deepcopy(c_old)
            neighbors = network.get_neighbors(i)
            c_prime[i] = random.choice(mycolor)
            # print("new color: ", c_prime[i])
            delta_U = delta_local_utility(network, i, neighbors, utility_matrix, c_old, c_prime, weight)
            p = min(1, np.exp(delta_U / temp))
            # print("p = ", p)
            if random.random() < p:
                # print("update")
                c[i] = int(c_prime[i])
            else:
                pass
            # print("current coloring: ", c)
        welfare_record.append(welfare(network, c, utility_matrix, weight))

    return c, welfare_record


def MH_policy_sync(network, mycolor, weight, utility, T = 2000, temp_type = 0, omega = 0.7):
    c = random_coloring(network, mycolor, seed = 20) # initialization
    # print(c)
    utility_matrix = utility
    welfare_record = [welfare(network, c, utility_matrix, weight)]
    for t in range(T):
        if temp_type == 0: # trigonometric additive cooling
            temp = 0.01 + 0.5 * (10 - 0.01)*(1 + np.cos(np.pi * t / T))
        if temp_type == 1: # multiplicative cooling
            temp = 10 * np.power(0.99, t)
        elif temp_type == 2: # logarithmical cooling
            temp = 0.1 / (1 + np.log(1 + 100 * t))
        elif temp_type == 3: # constant
            temp = 0.01
        c_old = deepcopy(c)
        # print("old color: ", c_old)
        # activated = activated_set(network, synchronic=True, omega= 0.5
        # omega= np.exp( - network.n * 100 / (temp * (network.n - 1)))
        activated = activated_set(network, synchronic=True, omega= omega)
        # print("omega = ", omega)
        # print("activated agents: ", activated)
        for i in activated:
            c_prime = deepcopy(c_old)
            #print("prime color: ", c_prime)
            neighbors = network.get_neighbors(i)
            c_prime[i] = random.choice(mycolor)
            #print("new color: ", c_prime[i])
            delta_U = delta_local_utility(network, i, neighbors, utility_matrix, c_old, c_prime, weight)
            p = min(1, np.exp(delta_U / temp))
            #print("p = ", p)
            if random.random() < p:
                #print("update")
                c[i] = int(c_prime[i])
            else:
                pass
        # print("current coloring: ", c)
        current_welfare = welfare(network, c, utility_matrix, weight)
        # print("current welfare: ", current_welfare)
        welfare_record.append(welfare(network, c, utility_matrix, weight))
    return c, welfare_record


def delta_local_loss(network, i, coloring, newcolor, utility, weight):
    new_coloring = deepcopy(coloring)
    new_coloring[i] = newcolor
    vertices_inrisk_old, prob_inrisk_old = network.get_inrisk(i, coloring)
    vertices_inrisk_new, prob_inrisk_new = network.get_inrisk(i, new_coloring)

    loss_after_i = (weight[i] * utility[i][newcolor] * (1 - isconflict(network, new_coloring, i))
                    * (1 - np.prod(np.array([1]*len(prob_inrisk_new) - np.array(prob_inrisk_new)))))
    loss_before_i = (weight[i] * utility[i][coloring[i]] * (1 - isconflict(network, coloring, i))
                     * (1 - np.prod(np.array([1]*len(prob_inrisk_old) - np.array(prob_inrisk_old)))))
    
    loss_after_rest = [] # omega * u
    for j in vertices_inrisk_new:
        loss_after_rest.append(weight[j] * utility[j][new_coloring[j]] * (1 - isconflict(network, new_coloring, j)))
    expected_loss_after_rest = np.dot(np.array(prob_inrisk_new), np.array(loss_after_rest))

    loss_before_rest = [] # omega * u
    for j in vertices_inrisk_old:
        loss_before_rest.append(weight[j] * utility[j][coloring[j]] * (1 - isconflict(network, coloring, j)))
    expected_loss_before_rest = np.dot(np.array(prob_inrisk_old), np.array(loss_before_rest))

    return loss_after_i + expected_loss_after_rest - loss_before_i - expected_loss_before_rest
                            
def MH_policy_async_RWO(network, c_WO1, mycolor, weight, utility, T = 2000, temp_type = 0):
    c = deepcopy(c_WO1) # initialization
    # print(c)
    utility_matrix = utility
    welfare_record = [welfare(network, c, utility_matrix, weight) - expected_loss(network, c, utility_matrix, weight)]

    for t in range(T):
        if temp_type == 0: # trigonometric additive cooling
            temp = 0.01 + 0.5 * (10 - 0.01)*(1 + np.cos(np.pi * t / T))
        if temp_type == 1: # multiplicative cooling
            temp = 10 * np.power(0.99, t)
        elif temp_type == 2: # logarithmical cooling
            temp = 1 / (1 + np.log(1 + 10 * t))
        elif temp_type == 3: # constant
            temp = 0.01
        random.seed(t)
        c_old = deepcopy(c)
        # print("old color: ", c_old)
        activated = activated_set(network, synchronic=False)
        # print("activated agents: ", activated)
        for i in activated:
            c_prime = deepcopy(c_old)
            neighbors = network.get_neighbors(i)
            c_prime[i] = random.choice(list(set(mycolor) - set(network.neighboring_colors(c_prime, i))))
            # print("new color: ", c_prime[i])
            
            delta_U = delta_local_utility(network, i, neighbors, utility_matrix, c_old, c_prime, weight)
            # print("delta_U = ", delta_U)
            delta_L = delta_local_loss(network, i, c_old, c_prime[i], utility_matrix, weight)
            # print("delta_L = ", delta_L)
            p = min(1, np.exp((delta_U - delta_L)/ temp))
            # print("p = ", p)
            '''
            after = welfare(network, c_prime, utility_matrix, weight)- expected_loss(network, c_prime, utility_matrix, weight)
            before = welfare(network, c_old, utility_matrix, weight)- expected_loss(network, c_old, utility_matrix, weight)
            print("difference = ", after - before)       
            p = min(1, np.exp((after - before)/ temp))
            print("p = ", p)
            '''

            if random.random() < p:
                # print("update")
                c[i] = int(c_prime[i])
            else:
                pass
            # print("current coloring: ", c)
        welfare_record.append(welfare(network, c, utility_matrix, weight) - expected_loss(network, c, utility_matrix, weight))
        
    return c, welfare_record

def MH_policy_sync_RWO(network, c_WO1, mycolor, weight, utility, T = 2000, temp_type = 0):
    c = deepcopy(c_WO1) # initialization
    # print(c)
    utility_matrix = utility
    welfare_record = [welfare(network, c, utility_matrix, weight) - expected_loss(network, c, utility_matrix, weight)]

    for t in range(T):
        if temp_type == 0: # trigonometric additive cooling
            temp = 0.01 + 0.5 * (10 - 0.01)*(1 + np.cos(np.pi * t / T))
        if temp_type == 1: # multiplicative cooling
            temp = 10 * np.power(0.99, t)
        elif temp_type == 2: # logarithmical cooling
            temp = 1 / (1 + np.log(1 + 100 * t))
        elif temp_type == 3: # constant
            temp = 0.01
        random.seed(t)
        c_old = deepcopy(c)
        # print("old color: ", c_old)
        activated = activated_set(network, synchronic=True, omega=1)
        # print("activated agents: ", activated)
        for i in activated:
            c_prime = deepcopy(c_old)
            neighbors = network.get_neighbors(i)
            c_prime[i] = random.choice(list(set(mycolor) - set(network.neighboring_colors(c_prime, i))))
            # print("new color: ", c_prime[i])
            
            delta_U = delta_local_utility(network, i, neighbors, utility_matrix, c_old, c_prime, weight)
            # print("delta_U = ", delta_U)
            delta_L = delta_local_loss(network, i, c_old, c_prime[i], utility_matrix, weight)
            # print("delta_L = ", delta_L)
            p = min(1, np.exp((delta_U - delta_L)/ temp))
            # print("p = ", p)
            '''
            after = welfare(network, c_prime, utility_matrix, weight)- expected_loss(network, c_prime, utility_matrix, weight)
            before = welfare(network, c_old, utility_matrix, weight)- expected_loss(network, c_old, utility_matrix, weight)
            print("difference = ", after - before)       
            p = min(1, np.exp((after - before)/ temp))
            print("p = ", p)
            '''

            if random.random() < p:
                # print("update")
                c[i] = int(c_prime[i])
            else:
                pass
            # print("current coloring: ", c)
        
        welfare_record.append(welfare(network, c, utility_matrix, weight) - expected_loss(network, c, utility_matrix, weight))

    return c, welfare_record

def update_coloring(old_coloring, i, c):
    new_coloring = deepcopy(old_coloring)
    new_coloring[i] = c
    return new_coloring

def tabu_search_WO(network, mycolor, weight, utility, T = 2000, maxtenure = 10):
    c = random_coloring(network, mycolor, seed = 41) # initialization
    t = 0
    t_prime = 0
    tabu_v = np.zeros(network.n)
    tabu_p = np.zeros((network.n, len(mycolor)))

    welfare_record = [welfare(network, c, utility, weight)]

    while t < T:
        # print("t = ", t)
        t_prime += 1
        c_prime = deepcopy(c)
        delta = -math.inf
        V_prime = None
        for i in range(network.n):
            if tabu_v[i] < t_prime+ 1:
                delta_star = -math.inf
                for color in list(set(mycolor) - set(network.neighboring_colors(c_prime, i))):
                    new_coloring = update_coloring(c_prime, i, color)
                    after = welfare(network, new_coloring, utility, weight)
                    before = welfare(network, c_prime, utility, weight)
                    delta_temp = after - before
                    if delta_temp > delta_star:
                        delta_star = delta_temp
                        c_star_i = color
                if delta_star > delta:
                    V_prime = i
                    c_star = c_star_i
                    delta = delta_star
            else:
                pass
        c_prime[V_prime] = c_star
        if delta > 0:
            c = deepcopy(c_prime)
            welfare_record.append(welfare(network, c, utility, weight))
            t = 0
            tabu_v[V_prime] = t_prime + random.randint(1, maxtenure)
            tabu_p[V_prime][c_star] = t_prime + random.randint(3, maxtenure)
        else:
            t += 1
            
    return c, welfare_record

def tabu_search_RWO(network, c_WO2, mycolor, weight, utility, T = 2000, maxtenure = 10):
    c = c_WO2 # initialization
    t = 0
    t_prime = 0
    tabu_v = np.zeros(network.n)
    tabu_p = np.zeros((network.n, len(mycolor)))

    welfare_record = [welfare(network, c, utility, weight) - expected_loss(network, c, utility, weight)]

    while t < T:
        # print("t = ", t)
        t_prime += 1
        c_prime = deepcopy(c)
        delta = -math.inf
        V_prime = None
        for i in range(network.n):
            if tabu_v[i] < t + 1:
                delta_star = -math.inf
                for color in list(set(mycolor) - set(network.neighboring_colors(c_prime, i))):
                    new_coloring = update_coloring(c_prime, i, color)
                    after = welfare(network, new_coloring, utility, weight)- expected_loss(network, new_coloring, utility, weight)
                    before = welfare(network, c_prime, utility, weight)- expected_loss(network, c_prime, utility, weight)
                    delta_temp = after - before
                    if delta_temp > delta_star:
                        delta_star = delta_temp
                        c_star_i = color
                if delta_star > delta:
                    V_prime = i
                    c_star = c_star_i
                    delta = delta_star
            else:
                pass
        c_prime[V_prime] = c_star
        if delta > 0:
            c = deepcopy(c_prime)
            welfare_record.append(welfare(network, c, utility, weight) - expected_loss(network, c, utility, weight))
            t = 0
            tabu_v[V_prime] = t_prime + random.randint(1, maxtenure)
            tabu_p[V_prime][c_star] = t_prime + random.randint(3, maxtenure)
        else:
            t += 1
    return c, welfare_record

                        


                    
