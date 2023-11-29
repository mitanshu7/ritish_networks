##Importing module, igraph
import networkx as nx
import time
import pickle

##Loading the graph
nx_graph = nx.read_graphml("Jaipur_street_network.graphml")

##Calculating betweenness centrality as an example
tic = time.time()
vertex_betweenness1 = nx.betweenness_centrality(nx_graph)
toc = time.time()

print(f"Time taken to calculate betweenness_centrality for a jaipur street network on a CPU using igraph is: \n")
print(toc-tic)


# Save list to a file using pickle
with open('jaipur_vbc_nx.pkl', 'wb') as file:
    pickle.dump(vertex_betweenness1, file)