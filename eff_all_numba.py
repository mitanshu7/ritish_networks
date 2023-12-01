import igraph as ig
from time import time
import pickle
import numpy as np
from numba import njit
from glob import glob
##Timing the program

start_tic = time()

##Function to use numba and find efficiency

@njit(parallel=True)
def find_eff(igraph_matrix):
    sp = 1.0/np.array(igraph_matrix)
    np.fill_diagonal(sp, 0)
    inner_inverse_sum = np.sum(sp)
    return inner_inverse_sum


##Timing the program

start_tic = time()

##Lists all files in the current working directory when the criteria is matched

file_names = glob('*.graphml') ##Lists all files with the graphml extension
file_names.sort() ##Sorting them alphabetically

for file in file_names:

    ##Placeholder for different files

    print("#########################################################################################################")
    print(f"Processing file {file}")


    ##Loading the graph from file

    G_ig = ig.Graph.Read(file, format="graphml")


    # Get the number of nodes (vertices)

    num_nodes = G_ig.vcount()
    print("Number of nodes: ", num_nodes)


    # Get the number of edges

    num_edges = G_ig.ecount()
    print("Number of edges: ", num_edges)


    #Calculating Meshedness coefficient(alpha)

    alpha = (num_edges - num_nodes + 1)/(2*num_nodes - 5)
    print("The value of messhedness coefficient(alpha) is : ", alpha)


    # Calculating Number of vertex pairs =  n(n-1)

    num_vertex_pairs = num_nodes*(num_nodes-1)
    print("Number of vertex pairs: ",num_vertex_pairs)


    # Calculating Number of edge pairs =  m(m-1)

    num_edge_pairs = num_edges*(num_edges-1)
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
    shortest_path_lengths_matrix = G_ig.distances(weights='weight', algorithm='dijkstra')
    toc = time()

    print(f"Time taken to calculate shortest_path_lengths_matrix using dijkstra algorithm is {toc - tic} seconds")



    #Calculating the sum of inverse for all values

    inverse_sum = find_eff(shortest_path_lengths_matrix)


    #Calculating the efficiency of the network

    efficiency = (1/num_vertex_pairs)*inverse_sum
    print("Efficiency of the network is: ", efficiency)

    ##Placeholder for different files
    print("#########################################################################################################")
    print()



##Timing the program

stop_toc = time()
print(f"The whole program took {stop_toc-start_tic} seconds to run")
