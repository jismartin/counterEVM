import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats
import re
from typing import Tuple


def extract_hours(duration_str: str) -> int:
    """
    Extract days and hours from a duration string and convert them to
    hours.

    Parameters:
        duration_str (str): Duration string with format 'XdYh'
          
    Returns:
        Tuple with the number of days and hours

    """
    days = 0
    hours = 0
    if 'd' in duration_str:
        days_match = re.search(r'(\d+)d', duration_str)
        if days_match:
            days = int(days_match.group(1))
    if 'h' in duration_str:
        hours_match = re.search(r'(\d+)h', duration_str)
        if hours_match:
            hours = int(hours_match.group(1))
    return days*8 + hours

def visualize_dag(G: nx.DiGraph, figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Visualize a Directed Acyclic Graph (DAG) with a hierarchical layout.

    Parameters:
    G (networkx.DiGraph): A directed acyclic graph.
    figsize (tuple): Figure size (width, height).
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The provided graph is not a Directed Acyclic Graph (DAG).")

    # Create a copy of the graph to avoid modifying the original
    G0=G.copy()

    # Compute a hierarchical layout for better visualization of DAGs
    lorder = list(nx.topological_sort(G0))
    
    # Compute progressive levels (PL)
    Plevels = {node: 0 for node in G0.nodes}
    for node in lorder[1:-1]:  # Skip start and end
        Plevels[node] = 1 + max((Plevels[p] for p in G0.predecessors(node)), default=0)
    Plevels[max(G0.nodes())] = max(Plevels.values()) + 1
    nx.set_node_attributes(G0, Plevels, "subset")
    pos = nx.drawing.layout.multipartite_layout(G0, subset_key='subset')

    # Draw the graph
    plt.figure(figsize=figsize)
    nx.draw(G0, pos, with_labels=True, node_color="lightblue", alpha=0.7, edge_color="gray", arrows=True)
    plt.show()

def pert_sample(optimistic: float, most_probable: float, pessimistic: float, elapsed_time: float =0, complete_percent: float =0,
                size: int =1000, lamb: float =4) -> np.ndarray:
    """
    Generate random samples from a PERT-like distribution.

    Parameters:
        optimistic (float): Optimistic estimate (O)
        most_probable (float): Most Probable estimate (M)
        pessimistic (float): Pessimistic estimate (P)
        elapsed_time (float): Elapsed time since the start of the activity
        size (int): Number of samples to generate
        lamb (float): Shape parameter (lower values = skewed distribution)

    Returns:
        numpy array of sampled durations

    """

    # Planned estimates
    a=optimistic
    b=most_probable
    c=pessimistic

    # Compute alpha and Beta parameters
    alpha = 1 + lamb * (b-a) / (c-a)
    beta = 1 + lamb * (c-b) / (c-a)

    # Generate samples from Beta distribution
    samples = stats.beta.rvs(alpha, beta, size=size)

    # Scale samples to the feasible range
    samples = a + samples * (c-a)
    return samples if complete_percent == 0 else elapsed_time + (1-complete_percent) * samples	
