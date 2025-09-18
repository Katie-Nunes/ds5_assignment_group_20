import networkx as nx
import random
import pandas as pd

#3.1
# calc_pagerank (graph)
# initialize_graph(k)
# recursively_add_nodes (k, M)

def main(k, M, N):
    G = nx.star_graph(k)
    Graph = build_ba_graph(G, M, N)
    print(max(Graph.nodes())) # Sanity Check

def preferential_attach_targets(G: nx.Graph, M: int) -> int:
    # Calculate total degree sum
    total_degree = sum(dict(G.degree()).values())

    # Calculate probabilities for each node
    nodes = list(G.nodes())
    degrees = [G.degree(node) for node in nodes]
    probabilities = [d/total_degree for d in degrees]

    # Add new node (k+1 where k is the current max node)
    new_node = max(G.nodes()) + 1
    G.add_node(new_node)

    # Select M targets using preferential attachment
    targets = random.choices(nodes, weights=probabilities, k=M)

    # Add edges to the selected targets
    for target in targets:
        G.add_edge(new_node, target)
    return new_node

def build_ba_graph(G: nx.Graph, M: int, N: int) -> nx.Graph:
    for _ in range(max(G.nodes()), N):
        preferential_attach_targets(G, M)
    return G

# 3.2
def load_csv(path: str) -> nx.DiGraph:
    df = pd.read_csv(path)
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df, edge_attr='weight', create_using=Graphtype)
    return G

if __name__ == "__main__":
    main(5, 4, 400)