import igraph as ig
from time import time
import pickle
import numpy as np


##Timing the program

start_tic = time()


##Loading the graph from file

G_ig = ig.Graph.Read("Dehradun_street_network.graphml",format="graphml")


# Get the number of nodes (vertices)

num_nodes = G_ig.vcount()
print("Number of nodes: ", num_nodes)


# Get the number of edges

num_edges = G_ig.ecount()
print("Number of edges: ", num_edges)


#Calculating Meshedness coefficient(alpha)

alpha = (num_edges - num_nodes + 1)/(2*num_nodes - 5)
print("The value of messhedness coefficient(alpha) is : ", alpha)


# Calculating Number of vertex pairs =  n(n-1)/2 

num_vertex_pairs = num_nodes*(num_nodes-1)/2.0
print("Number of vertex pairs: ",num_vertex_pairs)


# Calculating Number of edge pairs =  m(m-1)/2 

num_edge_pairs = num_edges*(num_edges-1)/2.0
print("Number of edge pairs: ",num_edge_pairs)


# Get the degrees of all vertices
degrees = G_ig.degree()

# Calculate the average degree
average_degree = sum(degrees) / len(degrees)
print("The average degree of the network is: ", average_degree)


#probability of connection
poc = average_degree / (num_nodes - 1)
print("The Probability of the connection in the network is: ", poc)


## Calculates shortest path lengths for given vertices in a graph. If None, all vertices will be considered.

tic = time()
shortest_path_lengths_matrix = G_ig.distances(algorithm='dijkstra')
toc = time()

print(f"Time taken to calculate shortest_path_lengths_matrix using dijkstra algorithm is {toc - tic} seconds")

# Save list of list to a file using pickle

with open('Dehradun_street_network_shortest_path_lengths_matrix.pkl', 'wb') as file:
    pickle.dump(shortest_path_lengths_matrix, file)


#Calculating the sum of inverse for all values

sp = 1.0/np.array(shortest_path_lengths_matrix)
np.fill_diagonal(sp, 0)
inner_inverse_sum = np.sum(sp)


#Calculating the efficiency of the network

efficiency = (1/num_vertex_pairs)*inner_inverse_sum
print("Efficiency of the network is: ", efficiency)


#Creating adjacency matrix

adj_matrix_G = G_ig.get_adjacency()

# Save list of list to a file using pickle

with open('Dehradun_street_network_adjacency_matrix.pkl', 'wb') as file:
    pickle.dump(adj_matrix_G, file)


##Timing the program

stop_toc = time()
print(f"The whole program took {stop_toc-start_tic} seconds to run")
