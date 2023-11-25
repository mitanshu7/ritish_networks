##Importing module, igraph
import igraph as ig
import time

##Loading the graph
tic = time.time()

ig_graph = ig.Graph.Read("Jaipur_street_network.graphml",format="graphml")

toc = time.time()

print("Time taken to load graph:")
print(toc-tic)

# Get the number of nodes (vertices)
tic = time.time()
num_nodes = ig_graph.vcount()
toc = time.time()
print("Number of nodes:", num_nodes)

print("Time taken to calculate num nodes")
print(toc-tic)

# Get the number of edges
tic = time.time()
num_edges = ig_graph.ecount()
toc = time.time()
print("Number of edges:", num_edges)

print("Time taken to calculate num edges")
print(toc-tic)
