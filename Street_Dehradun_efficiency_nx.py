#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx



# In[2]:


#Reading the graphml files
G = nx.read_graphml("/home/ritishkhetarpal/ms_thesis/Dehradun/Dehradun_street_network.graphml")


# In[3]:


#Checking the number of nodes and edges
print("num of nodes = ", G.number_of_nodes())
print("num of edges = ", G.number_of_edges())


# In[ ]:


#Creating adjacency matrix
adj_matrix_G = nx.to_numpy_array(G)


# In[6]:


max_value = max(max(row) for row in adj_matrix_G)
print("Maximum value in the adjacency matrix:", max_value)


# In[ ]:


#Calculating Meshedness coefficient(alpha)
n = 58839   #nodes
k = 67925   #links
alpha = (k - n + 1)/(2*n - 5)
print("The value of messhedness coefficient(alpha) is : ", alpha)


# In[15]:


#Calculating the pairs of all shortest path length from dijkstra algorithm 
shortest_path_lengths_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))


# In[16]:


#Calculating the sum of inverse for all values in all nested dict
inner_inverse_sum = 0
for outer_key, inner_dict in shortest_path_lengths_dict.items():
    for inner_values in inner_dict.values():
        if inner_values != 0:
            values = 1/inner_values
            inner_inverse_sum += values


# In[17]:


#calculating the value of n which should be equal to number of nodes here. 
n= len(shortest_path_lengths_dict)
print("Number of nodes: ", n)
# Calculating Number of pairs = n(n-1) not n(n-1)/2 because in dict we are considering twice
num_pairs = n*(n-1)
print("Number of pairs: ",num_pairs)


# In[18]:


#Calculating the efficiency of the network
Efficiency = (1/num_pairs)*inner_inverse_sum
print("Efficiency of the network is:", Efficiency)


# In[22]:


#Average Degree
total_degree = sum(dict(G.degree()).values())
av_degree = total_degree / len(G.nodes())
print("The average degree of the network is: ", av_degree)

#probability of connection
p = av_degree / (len(G.nodes()) - 1)
print("The Probability of the connection in the network is: ", p)

