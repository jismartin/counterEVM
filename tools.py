import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from typing import Tuple


# Read a project file in Patterson's format and create the Directec Acyclic Graph (DAG)
def read_dag_file(file_path: str) -> nx.DiGraph:
    """
    Read a project file in Patterson's format and create the Directec Acyclic Graph (DAG).
    In Vanhoucke's dataset, the first line and the fourth line are empty, and empty lines are used to separate the data.
    
    Args:
        file_path (str): The path to the project file.

    Returns:
        nx.DiGraph: A directed acyclic graph representing a project network.
    """
    G=nx.DiGraph()
    try:
        with open(file_path, 'r') as file:
            # Fisrt line is empty
            file.readline()
            # Read number of activities and resources
            num_activities, num_resources = map(int, file.readline().strip().split())
            # Read resource availability
            resource_availability = list(map(int, file.readline().strip().split()))
            # Fourth line is empty
            file.readline()
            # Read data for each activity
            for activity_id in range(1, num_activities + 1):
                data = file.readline().strip().split()
                duration = int(data[0])
                G.add_node(activity_id, duration=duration)
                required_resources = list(map(int, data[1:1 + num_resources]))
                num_successors = int(data[1 + num_resources])
                for successor in data[2 + num_resources:]:
                    G.add_edge(activity_id, int(successor))
        return G

    except Exception as e:
        print(f"Error reading the file: {e}")
        return None


# Compute the topological indicators for a given network (Vanhoucke, 2008)
def compute_topological_indicators(G: nx.DiGraph) -> Tuple[float, float, float, float]:
    """
    Computes four topological indicators proposed by Vanhoucke of a directed graph.

    Args:
        G (nx.DiGraph): A directed acyclic graph representing a project network. 
        The start and end nodes are dummy activities.

    Returns:
        tuple[float, float, float, float]: A tuple containing four topological 
        indicators: the serial-parallel (SP), the activity distribution (AD),
        the short arcs (LA), and the topological flow (TF).
    """
    n = G.number_of_nodes() - 2  # Excluding start and end
    lnodes = list(G.nodes()) 
    start, end = lnodes[0], lnodes[-1] # Start and end nodes are dummy activities
    lorder = list(nx.topological_sort(G))
    
    # Compute progressive levels (PL)
    Plevels = {node: 0 for node in G.nodes}
    for node in lorder[1:-1]:  # Skip start and end
        Plevels[node] = 1 + max((Plevels[p] for p in G.predecessors(node)), default=0)

    Plevels.pop(start)
    Plevels.pop(end)

    m = max(Plevels.values(), default=0)

    # Compute regressive levels (RL)
    Rlevels = {node: m + 1 for node in G.nodes}
    for node in list(reversed(lorder))[1:-1]:  # Reverse order, skip start and end
        Rlevels[node] = min((Rlevels[s] for s in G.successors(node)), default=m) - 1

    Rlevels.pop(start)
    Rlevels.pop(end)

    # Count occurrences of each progressive level
    Wlevels_counts = Counter(Plevels.values())
    Wlevels = np.array(list(Wlevels_counts.values()))
    D = np.sum(Wlevels[:-1] * Wlevels[1:]) if len(Wlevels) > 1 else 0

    # Serial-parallel indicator (I2-SP)
    SP = (m - 1) / (n - 1) if n > 1 else 1

    # Activity distribution indicator (I3-AD)
    W_mean = Wlevels.mean() if len(Wlevels) > 0 else 0
    AD = np.abs(Wlevels - W_mean).sum() / (2 * (m - 1) * (W_mean - 1)) if (m > 1) and (m < n) else 0

    # Compute short arcs indicator (I4-LA)
    lengths_arcs = [Plevels[j] - Plevels[i] for i, j in G.edges if i != start and j != end]
    lengths_counts = Counter(lengths_arcs)
    lengths = np.array(list(lengths_counts.values()))
    LA = (lengths[0] - n + Wlevels[0]) / (D - n + Wlevels[0]) if D != (n - Wlevels[0]) else 1

    # Topological flow indicator (I6-TF)
    TF = np.sum(np.fromiter((Rlevels[node] - Plevels[node] for node in Plevels), dtype=int)) / ((m - 1) * (n - m)) if (m > 1) and (m < n) else 0

    return SP, AD, LA, TF


def visualize_dag(G: nx.DiGraph, figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Visualize a Directed Acyclic Graph (DAG) with a hierarchical layout.

    Parameters:
    G (networkx.DiGraph): A directed acyclic graph.
    figsize (tuple): Figure size (width, height).
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The provided graph is not a Directed Acyclic Graph (DAG).")

    # Compute a hierarchical layout for better visualization of DAGs
    lorder = list(nx.topological_sort(G))
    
    # Compute progressive levels (PL)
    Plevels = {node: 0 for node in G.nodes}
    for node in lorder[1:-1]:  # Skip start and end
        Plevels[node] = 1 + max((Plevels[p] for p in G.predecessors(node)), default=0)
    Plevels[max(G.nodes())] = max(Plevels.values()) + 1
    nx.set_node_attributes(G, Plevels, "subset")
    pos = nx.drawing.layout.multipartite_layout(G, subset_key='subset')

    # Draw the graph
    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", alpha=0.7, edge_color="gray", arrows=True)
    plt.show()


def compute_degree_entropy(G: nx.DiGraph, type: str ='in') -> float:
    """
    Compute the degree entropy of a Directed Acyclic Graph (DAG).

    Args:
        G (nx.DiGraph): A directed acyclic graph.
        type (str): The type of degree to compute the entropy. It can be 'in' or 'out'.

    Returns:
        float: The degree entropy of the DAG.
    """

    if type=='in':
        degrees=[k for _, k in G.in_degree()]
    elif type=='out':
        degrees=[k for _, k in G.out_degree()]
    else:
        raise ValueError('type must be in or out')
    
    k, n_k = np.unique(degrees, return_counts=True)
    n_k = n_k / n_k.sum()
    entropy = -np.sum(n_k * np.log(n_k))
    return entropy


def compute_betweenness_heterogeneity(G: nx.DiGraph) -> float:
    """
    Compute the betweenness heterogeneity of a Directed Acyclic Graph (DAG).

    Args:
        G (nx.DiGraph): A directed acyclic graph.

    Returns:
        float: The betweenness heterogeneity of the DAG.
    """

    betweenness = [b for b in nx.betweenness_centrality(G, normalized=False).values()]
    return np.sum(np.array(betweenness)**2) / np.mean(betweenness)**2


def compute_modularity(G: nx.Graph) -> float:
    """
    Compute the modularity of the best partition of a graph using Louvain algorithm (assuming undirected graph).

    Args:
        G (nx.Graph): An undirected graph.

    Returns:
        float: The modularity of the best partition of the graph.
    """
    
    # Partition the graph using Louvain algorithm (assuming undirected graph)
    partition = nx.algorithms.community.louvain_communities(G)
    return nx.algorithms.community.modularity(G, partition)


def compute_average_path_length(G: nx.DiGraph) -> float:
    """
    Compute the average path length of all paths between the first and the last node.

    Args:
        G (nx.DiGraph): A directed acyclic graph.

    Returns:
        float: The average path length of all paths between the first and the last node.
    """

    return np.mean( [len(p) for p in nx.all_simple_paths(G, 1, G.number_of_nodes())])