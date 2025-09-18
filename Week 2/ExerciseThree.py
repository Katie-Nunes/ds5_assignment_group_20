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

def random_walk_pagerank(G: nx.DiGraph | nx.DiGraph, 
                         alpha: float = 0.85, 
                         steps: int = 3000_000
                         start: Hashable | None = None,
                         seed: int | None = None,
                            ) -> dict[Hashable, float]:
## Aaron Randm walk pagerank with teleportation 
##    Exercise 3.1 + 3.2 in this one block line of code
    if seed is not None:
        random.seed(seed)

    nodes = list(G.nodes())
    if not nodes:
        return {}
# Pick starting node 
    current = start if (start in G) else random.choice(nodes)
    # Track number of visits to each node.
    # At the end, dividing visits by steps gives PageRank estimate.
    visits: Dict[Hashable, int] = {node: 0 for node in nodes}

    for _ in range(steps):
        visits[current] += 1
        follow_link = (random.random() < alpha)
        if follow_link
            # 3.1 undirected 
            if not G.is_directed():
                nbrs = list(G.neighbors(current))
            # 3.2 directedd
            else:
                nbrs = list(G.successors(current))
            if nbrs:
                current = random.choice(nbrs)
                continue
            # if no neighbors fall through to teleport
        current = random.choice(nodes)

    # Teleport (used in both cases 3.1 and 3.2)
    total = float(sum(visits.values()))
    return {n: visits[n] / total for n in nodes}
    
        


# 3.2
def load_csv(path: str) -> nx.DiGraph:
    df = pd.read_csv(path)
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df, edge_attr='weight', create_using=Graphtype)
    return G

if __name__ == "__main__":
    main(5, 4, 400)