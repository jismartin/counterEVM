import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats
import re
import os
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


# Draw random durations for the activities following a PERT distribution
def draw_random_durations(G,control=None,pert_sample=pert_sample):
    nx.set_node_attributes(G,0,'duration')
    if control is None:
        for n in list(G.nodes())[1:-1]: # Skip the start and end nodes
            G.nodes[n]['duration']=pert_sample(G.nodes[n]['optimistic'], G.nodes[n]['mostlikely'], G.nodes[n]['pessimistic'],size=1)[0]
    else:
        for n in control.index[1:-1]: # Skip the start and end nodes
          
            if control.loc[n,'PercentageCompleted']==1: # completed activity
                G.nodes[n]['duration']=control.loc[n,'ActualDuration'] # the duration is the actual duration
            
            elif control.loc[n,'PercentageCompleted']>0 and control.loc[n,'PercentageCompleted'] <1: # ongoing activity
                G.nodes[n]['duration']=pert_sample(G.nodes[n]['optimistic'], G.nodes[n]['mostlikely'], G.nodes[n]['pessimistic'],
                                                   elapsed_time=control.loc[n,'ActualDuration'],
                                                   complete_percent=control.loc[n,'PercentageCompleted'],
                                                   size=1)[0]
            else: # not started activity
                G.nodes[n]['duration']=pert_sample(G.nodes[n]['optimistic'], G.nodes[n]['mostlikely'], G.nodes[n]['pessimistic'],size=1)[0]


# Compute the start time of activities based on the dependencies
def compute_times(G,time='duration'):
    # time: duration or meanduration
    start_time={n:0 for n in G.nodes()} # initialize the start time dictionary
    finish_time={n:0 for n in G.nodes()} # initialize the finish time dictionary
    for n in nx.topological_sort(G): # follow a list of nodes in topologically sorted order
        end=start_time[n] + G.nodes[n][time] # end time
        for j in G.neighbors(n):
            if G.edges[n,j]['type']=='FS': # Finish to Start dependency
                start_time[j]=np.max([start_time[j],end])
            if G.edges[n,j]['type']=='SS': # Start to Start dependency
                start_time[j]=np.max([start_time[j],start_time[n]])
    for n in G.nodes():
        finish_time[n]=start_time[n]+G.nodes[n][time]
    return start_time,finish_time

def find_critical_path(G, time='duration'):
    # Compute the start and finish times
    start_time, finish_time = compute_times(G, time)  # Assume function is correctly implemented

    # Identify the unique end node using topological sorting
    topo_order = list(nx.topological_sort(G))
    end_node = topo_order[-1]  # The last node in topological order is the end node

    # Trace back from the end node using reverse topological order
    critical_path = [end_node]
    current = end_node
    while True:
        predecessors = list(G.predecessors(current))
        if not predecessors:
            break
        # Choose the predecessor that has the latest finish time that matches its successor's start time
        critical_predecessor = max(predecessors, key=lambda p: finish_time[p])
        if finish_time[critical_predecessor] != start_time[current]:
            break
        critical_path.append(critical_predecessor)
        current = critical_predecessor

    critical_path.reverse()  # Reverse to get the correct order
    return critical_path

# Monte-Carlo simulation of the project
def simulation(G,project_name,experiment,control=None,Nruns=10000,echo=False):
    sim_list=list() # list of dictionaries with runs
    if echo: print('Starting simulation...')
    for m in range(Nruns):
        # Draw durations
        draw_random_durations(G,control)
        # Save run
        sim={'duration'+str(i):G.nodes[i]['duration'] for i in G.nodes()}
        sim.update({'critical_path':find_critical_path(G,time='duration')})
        sim.update({'baseline_duration':np.max([t for t in compute_times(G,time='mean_duration')[1].values()])}) # baseline duration based on mean durations
        sim.update({'actual_duration':np.max([t for t in compute_times(G,time='duration')[1].values()])}) # final duration based on actual durations
        sim_list.append(sim)
        if echo: print(m)
    if echo: print('... end simulation')

    if not os.path.exists('./data/'+project_name):
        os.makedirs('./data/'+project_name)
    pd.DataFrame.from_dict(sim_list).to_csv('./data/'+project_name+'/simulation_'+project_name+'_' + experiment+ '.csv')
