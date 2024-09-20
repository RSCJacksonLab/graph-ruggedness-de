import graph_ruggedness_de
import numpy as np
import networkx as nx
import math
from scipy.stats import norm

def compute_log_probability(prior_arr: list, 
                            empirical_val: float,
                            epsilon: float = 1e-300):

    '''
    Function to compute the log probability of sampling an empirical
    ruggedness value from a prior array of expected ruggedness values.

    Arguments:
    ----------
    prior_arr: list
        List of expected values determined by Brownian diffusion. 
    
    empirical_val: float
        The empirical value without diffusion. 
    
    epsilon: float, defaut=`1e-300`
        A constant added for numerical stability.
    '''

    mean = np.mean(prior_arr)
    std_dev = np.std(prior_arr)
    probability_density = norm.pdf(empirical_val, mean, std_dev)
    return np.log(probability_density+epsilon)


def parameterise_sample_size(G,
                             min_sample_size, 
                             max_sample_size, 
                             step):
    
    '''
    
    '''
    
    probabilities = []
    for sample_size in np.arange(min_sample_size, max_sample_size, step):
        G_n_eigh = graph_ruggedness_de.compute_elementary_landscape(G=G, n=5)
        dir_eigh = graph_ruggedness_de.compute_dirichlet_energy(G=G_n_eigh)

        prior_dist = sample_prior_dist(G, 
                                       sample_size=sample_size, 
                                       ruggedness_fn=graph_ruggedness_de.compute_dirichlet_energy,
                                       replicates=10)
        mean = np.mean(prior_dist)
        std_dev = np.std(prior_dist)
        probabilities.append(norm.pdf(dir_eigh, mean, std_dev))

    max_val = max(x for x in probabilities if not math.isnan(x))
    optimal_idx = probabilities.index(max_val)
    
    return np.arange(min_sample_size, max_sample_size, step)[optimal_idx], probabilities[optimal_idx]

def sample_prior_dist(G, 
                      ruggedness_fn, 
                      sample_size_range: tuple = (0.2, 0.6),
                      sample_size: float = None,
                      replicates=100, 
                      _fluctuation_scale=0,
                      local: bool =False,):

    '''
    
    '''

    prior_dist = []
    local_dir_dict = {}
    for _, _ in enumerate(range(replicates)):

        G_cpy = G.copy()
        
        if sample_size is not None:
            sampled_size = sample_size
        else:
            sampled_size = np.random.uniform(sample_size_range[0], sample_size_range[1])
        
        G_sampled = sample_values(G=G_cpy, sample_size=sampled_size)
        
        G_diffused = brownian_diffusion(G=G_sampled, fluctuation_scale=_fluctuation_scale)

        if not local:
            prior_dist.append(ruggedness_fn(G_diffused))
        else:
            graph_ruggedness_de.compute_local_dirichlet_energy(G=G_diffused)
            for node in G_sampled.nodes():
                if node in local_dir_dict.keys():
                    local_dir_dict[node] += [G_diffused.nodes[node]['local_dirichlet']]
                else:
                    local_dir_dict[node] = [G_diffused.nodes[node]['local_dirichlet']]

    
    if not local:
        prior_arr = np.array(prior_dist)
        return prior_arr[~np.isnan(prior_arr)] #Mask out any NaN values 
    else:
        return local_dir_dict
    
def brownian_diffusion(G, fluctuation_scale):

    '''
    
    '''
    G = G.copy()

    if not nx.is_connected(G):
        raise Exception('More than a single connected component in the graph. All nodes must be connected.')

    iterations = len(G) ** 2

    for _ in range(iterations):

        values = [G.nodes[node]['value'] for node in G.nodes()]
        if not np.any(np.isnan(values)):
            break

        new_values = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            avg_neighbor_signal = np.nanmean([G.nodes[neighbor]['value'] for neighbor in neighbors])
            fluctuation = np.random.randn() * fluctuation_scale
            
            if np.isnan(G.nodes[node]['value']):
                new_values[node] = avg_neighbor_signal + fluctuation
            else:
                new_values[node] = (G.nodes[node]['value'] + avg_neighbor_signal) / 2 + fluctuation
        
        for node in new_values:
            G.nodes[node]['value'] = new_values[node]
    
    return G

def sample_values(G: nx.Graph,
                  sample_size: float):
    '''
    
    '''
    
    G = G.copy()
    num_nodes = G.number_of_nodes()
    num_sampled = int(sample_size * num_nodes)

    if num_sampled <= 2:
        raise ValueError('< 2 nodes sampled in subgraph. Increase sample size.')

    values = np.array([G.nodes[node]['value'] for node in G])
    max_value = values.max()
    min_value = values.min()
    eligible_nodes = [node for node in G if G.nodes[node]['value'] not in (max_value, min_value)]
    num_sampled = min(num_sampled - 2, len(eligible_nodes))
    sampled_nodes = np.random.choice(eligible_nodes, size=num_sampled, replace=False).tolist()

    for node in G.nodes():
        if node not in sampled_nodes:
            G.nodes[node]['value'] = np.nan

    return G 