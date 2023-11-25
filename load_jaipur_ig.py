##Importing module, igraph
import igraph as ig
import time
import pickle

##Loading the graph
ig_graph = ig.Graph.Read("Jaipur_street_network.graphml",format="graphml")

##Calculating betweenness centrality as an example
tic = time.time()
vertex_betweenness1 = ig_graph.betweenness()
toc = time.time()

print(f"Time taken to calculate betweenness_centrality for a jaipur street network on a CPU using igraph is: \n")
print(toc-tic)


# Save list to a file using pickle
with open('jaipur_vbc.pkl', 'wb') as file:
    pickle.dump(vertex_betweenness1, file)
