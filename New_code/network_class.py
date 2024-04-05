import numpy as np;
import networkx as nx;
import matplotlib.pyplot as plt;
import random
import copy
import math
from copy import copy, deepcopy
from itertools import product

class network:
    def __init__(self, n, prob, seed = 42):
        self.n = n
        self.prob = prob
        self.G = nx.erdos_renyi_graph(n, prob, seed = seed)

    def get_delta(self):
        degrees = [val for (node, val) in self.G.degree()]
        delta = max(degrees)
        return delta
    
    def n_color(self):
        return self.get_delta() + 1
        
    def convert_to_numpy(self):
        A_0 = np.asarray(nx.to_numpy_array(self.G))
        A = A_0[:] # convert to numpy array
        return A
    
    def plot(self):
        # Create a 3D figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the 3D graph
        pos = nx.spring_layout(self.G, dim=3)
        for edge in self.G.edges():
            u, v = edge
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]], 'k-')
            
        # Plot the nodes
        for node in self.G.nodes():
            x, y, z= pos[node]
            ax.scatter(x, y, z, c='b', s=100, label='Node {}'.format(node))

        ax.set_axis_off()
        # ax.legend()
    
        # Save the 3D graph
        plt.savefig('network_3D.png', bbox_inches='tight')
        plt.show()

    def get_neighbors(self, i):
        return list(self.G.neighbors(i))
    
    def neighboring_colors(self, coloring, i):
        neighbors = self.get_neighbors(i)
        return list(set([coloring[j] for j in neighbors]))
    
    def get_family(self, coloring, i, info2 = False):
        neighbors = self.get_neighbors(i)
        if not info2:
            return neighbors
        else:
            members = [index for index, element in enumerate(coloring) if element == coloring[i]]
            members.remove(i)
            return set(neighbors + members)
        
    def complementary_edges(self, bidirected = False):
        if not bidirected:
            edges = list(self.G.edges())
            edges = [(i, j) if i < j else (j, i) for (i, j) in edges]
            all_edges = product(range(self.n), range(self.n))
            all_edges = [(i, j) for (i, j) in all_edges if i < j]
            return list(set(all_edges) - set(edges))
        else:
            edges = list(self.G.edges())
            edges = [(i, j) if i < j else (j, i) for (i, j) in edges]
            edges = set(edges)
            all_edges = set(product(range(self.n), range(self.n)))
            return list(all_edges - edges)
    
    def complementary_prob(self, bidirected = False):
        # the probability of connecting a complementary edge is random
        complementary_prob = [0] * len(self.complementary_edges(bidirected = bidirected))
        for k in range(len(complementary_prob)):
            random.seed(k)
            complementary_prob[k] = random.random()
        return complementary_prob
    
    def get_inrisk(self, i, coloring, bidirected = False): # get the probability of connecting an $e \in \CF_i^c / \CN(V_i)$
        complementary_edges = self.complementary_edges(bidirected=bidirected)
        random.seed(42)
        complementary_prob = self.complementary_prob(bidirected=bidirected)
        complementary_inrisk = [(a, b) for _, (a, b) in enumerate(complementary_edges) if (a == i or b == i) and (coloring[a] == coloring[b])]
        vertices_inrisk = [a if a != i else b for (a, b) in complementary_inrisk]
        indices_of_complementary_inrisk = [idx for idx, (a, b) in enumerate(complementary_edges) if (a == i or b == i) and (coloring[a] == coloring[b])]
        prob_of_complementary_inrisk = [complementary_prob[idx] for idx in indices_of_complementary_inrisk]
        return vertices_inrisk, prob_of_complementary_inrisk
        



            
    
    

    