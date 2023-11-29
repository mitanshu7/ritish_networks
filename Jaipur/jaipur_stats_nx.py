##Importing module, igraph
import networkx as nx
import time
import psutil

##Compute node stats
print("Ram usage at start: ",psutil.virtual_memory())
print("RAM precentage: ",psutil.virtual_memory().percent)
print("CPU usage: ", psutil.cpu_percent())

##Loading the graph

tic = time.time()

nx_graph = nx.read_graphml("Jaipur_street_network.graphml")

toc = time.time()

print("Time taken to load graph:")
print(toc-tic)

##Compute node stats
print("Ram usage after loading graph/before calculating num_nodes : ",psutil.virtual_memory())
print("RAM precentage: ",psutil.virtual_memory().percent)
print("CPU usage: ", psutil.cpu_percent())

# Get the number of nodes (vertices)
tic = time.time()
num_nodes = nx_graph.number_of_nodes()
toc = time.time()
print("Number of nodes:", num_nodes)

print("Time taken to calculate num nodes")
print(toc-tic)

##Compute node stats
print("Ram usage after calculating num_nodes/before calculating num_edges : ",psutil.virtual_memory())
print("RAM precentage: ",psutil.virtual_memory().percent)
print("CPU usage: ", psutil.cpu_percent())

# Get the number of edges
tic = time.time()
num_edges = nx_graph.number_of_edges()
toc = time.time()
print("Number of edges:", num_edges)

print("Time taken to calculate num edges")
print(toc-tic)

##Compute node stats
print("Ram usage at end: ",psutil.virtual_memory())
print("RAM precentage: ",psutil.virtual_memory().percent)
print("CPU usage: ", psutil.cpu_percent())