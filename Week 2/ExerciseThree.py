import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import Hashable,Dict
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
    targets = []
    available_nodes = nodes.copy()
    available_probs = probabilities.copy()

    for _ in range(M):
        if not available_nodes:
            break
        target = random.choices(available_nodes, weights=available_probs, k=1)[0]
        targets.append(target)
        
        # Remove the selected node to avoid duplicate links
        idx = available_nodes.index(target)
        available_nodes.pop(idx)
        available_probs.pop(idx)
        
        # Renormalize probabilities
        total = sum(available_probs)
        if total > 0:
            available_probs = [p/total for p in available_probs]

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
     steps: int = 100000,
     start: Hashable = None,
     seed: int = None) -> Dict[Hashable, float]:
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
        if follow_link:
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
    
def draw_graph_with_scores(G: nx.Graph | nx.DiGraph, 
                          scores: Dict[Hashable, float], 
                          title: str = "") -> None:
    """
    Visualize graph with node sizes proportional to PageRank scores.
    
    Parameters:
    -----------
    G : nx.Graph or nx.DiGraph
        The network graph to visualize
    scores : Dict[Hashable, float]
        PageRank scores for each node
    title : str, optional
        Title for the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Scale node sizes for better visualization
    node_sizes = [scores[node] * 50000 + 100 for node in G.nodes()]
    
    # Choose layout based on graph type and size
    if len(G.nodes()) <= 50:
        pos = nx.spring_layout(G, seed=42)
    else:
        # For larger graphs, use a faster layout
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=20, seed=42)
    
    # Draw the graph
    if isinstance(G, nx.DiGraph):
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrowstyle='->', 
                              arrowsize=10, edge_color='gray', width=0.5)
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', width=0.5)
    
    # Draw nodes with sizes proportional to PageRank
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                  node_color=list(scores.values()),
                                  cmap='viridis', alpha=0.8)
    
    # Add labels for smaller graphs
    if len(G.nodes()) <= 30:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.colorbar(nodes, label='PageRank Score')
    plt.title(f"{title}\nNode size ∝ PageRank")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_distribution(scores: Dict[Hashable, float], title: str = "") -> None:
    """
    Plot the distribution of PageRank values.
    
    Parameters:
    -----------
    scores : Dict[Hashable, float]
        PageRank scores for each node
    title : str, optional
        Title for the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Extract scores and sort them
    values = list(scores.values())
    values.sort(reverse=True)
    
    # Create plots
    plt.subplot(1, 2, 1)
    plt.hist(values, bins=30, alpha=0.7, edgecolor='black', density=True)
    plt.xlabel('PageRank Value')
    plt.ylabel('Density')
    plt.title('Distribution of PageRank Values')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.loglog(range(1, len(values) + 1), values, 'o', alpha=0.7, markersize=4)
    plt.xlabel('Rank (log)')
    plt.ylabel('PageRank Value (log)')
    plt.title('Rank-Value Plot (Log-Log)')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def analyze_network_properties(G: nx.Graph | nx.DiGraph, 
                             scores: Dict[Hashable, float]) -> Dict[str, Any]:
    """
    Analyze and return network properties for the notebook.
    
    Parameters:
    -----------
    G : nx.Graph or nx.DiGraph
        The network graph
    scores : Dict[Hashable, float]
        PageRank scores
        
    Returns:
    --------
    Dict with network properties
    """
    pagerank_values = list(scores.values())
    
    properties = {
        'num_nodes': len(G.nodes()),
        'num_edges': len(G.edges()),
        'is_directed': isinstance(G, nx.DiGraph),
        'avg_pagerank': np.mean(pagerank_values),
        'max_pagerank': max(pagerank_values),
        'min_pagerank': min(pagerank_values),
        'pagerank_std': np.std(pagerank_values),
        'degree_assortativity': nx.degree_assortativity_coefficient(G) if not isinstance(G, nx.DiGraph) else None,
    }
    
    # Handle clustering coefficient calculation
    try:
        if not isinstance(G, nx.DiGraph):
            properties['avg_clustering'] = nx.average_clustering(G)
        else:
            properties['avg_clustering'] = nx.average_clustering(G.to_undirected())
    except:
        properties['avg_clustering'] = 'N/A'
    
    return properties

def create_summary_table(properties_ba: Dict[str, Any], 
                        properties_squirrel: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary table comparing both networks.
    
    Parameters:
    -----------
    properties_ba : Dict
        Properties of Barabási-Albert network
    properties_squirrel : Dict
        Properties of squirrel network
        
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    data = {
        'Property': ['Number of Nodes', 'Number of Edges', 'Directed', 
                    'Average PageRank', 'Max PageRank', 'Min PageRank',
                    'PageRank STD', 'Degree Assortativity', 'Average Clustering'],
        'Barabási-Albert': [
            properties_ba['num_nodes'],
            properties_ba['num_edges'],
            properties_ba['is_directed'],
            f"{properties_ba['avg_pagerank']:.6f}",
            f"{properties_ba['max_pagerank']:.6f}",
            f"{properties_ba['min_pagerank']:.6f}",
            f"{properties_ba['pagerank_std']:.6f}",
            f"{properties_ba['degree_assortativity']:.4f}" if properties_ba['degree_assortativity'] is not None else 'N/A',
            f"{properties_ba['avg_clustering']:.4f}" if isinstance(properties_ba['avg_clustering'], float) else properties_ba['avg_clustering']
        ],
        'Squirrel Network': [
            properties_squirrel['num_nodes'],
            properties_squirrel['num_edges'],
            properties_squirrel['is_directed'],
            f"{properties_squirrel['avg_pagerank']:.6f}",
            f"{properties_squirrel['max_pagerank']:.6f}",
            f"{properties_squirrel['min_pagerank']:.6f}",
            f"{properties_squirrel['pagerank_std']:.6f}",
            f"{properties_squirrel['degree_assortativity']:.4f}" if properties_squirrel['degree_assortativity'] is not None else 'N/A',
            f"{properties_squirrel['avg_clustering']:.4f}" if isinstance(properties_squirrel['avg_clustering'], float) else properties_squirrel['avg_clustering']
        ]
    }
    
    return pd.DataFrame(data)     


# 3.2
def load_csv(path: str) -> nx.DiGraph:
    df = pd.read_csv(path)
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df, edge_attr='weight', create_using=Graphtype)
    return G

if __name__ == "__main__":
    print("=== Exercise 3.1: Barabási-Albert Network ===")
    
    # Generate BA graph
    G_ba = main(5, 4, 100)  # Using 100 nodes for faster demo
    pagerank_ba = random_walk_pagerank(G_ba, alpha=0.85, steps=50000, seed=42)
    
    # Visualize BA graph
    draw_graph_with_scores(G_ba, pagerank_ba, "Barabási-Albert Network (Undirected)")
    plot_distribution(pagerank_ba, "Barabási-Albert Network PageRank Distribution")
    
    print("\n=== Exercise 3.2: Squirrel Network ===")
    try:
        # Load squirrel graph
        G_squirrel = load_csv("squirrel_edges.csv")
        pagerank_squirrel = random_walk_pagerank(G_squirrel, alpha=0.85, steps=50000, seed=42)
        
        # Visualize squirrel graph
        draw_graph_with_scores(G_squirrel, pagerank_squirrel, "Squirrel Network (Directed)")
        plot_distribution(pagerank_squirrel, "Squirrel Network PageRank Distribution")
        
        # Analyze properties
        props_ba = analyze_network_properties(G_ba, pagerank_ba)
        props_squirrel = analyze_network_properties(G_squirrel, pagerank_squirrel)
        
        # Create summary table
        summary_df = create_summary_table(props_ba, props_squirrel)
        print("\n=== Network Comparison Summary ===")
        print(summary_df.to_string(index=False))
        
    except FileNotFoundError:
        print("Squirrel edges CSV file not found. Skipping Exercise 3.2.")
    except Exception as e:
        print(f"Error loading squirrel data: {e}")