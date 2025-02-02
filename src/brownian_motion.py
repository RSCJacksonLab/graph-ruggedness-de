import graph_ruggedness_de
import numpy as np
import networkx as nx
import math
from scipy.stats import norm
from scipy import stats
from joblib import Parallel, delayed

def compute_log_probability(prior_arr: list, 
                            empirical_val: float,):

    '''
    Function to compute the log probability of sampling an empirical
    ruggedness value from a prior array of expected ruggedness values.

    Arguments:
    ----------
    prior_arr: list
        List of expected values determined by Brownian diffusion. 
    
    empirical_val: float
        The empirical value without diffusion. 

    '''

    mean = np.mean(prior_arr)
    std_dev = np.std(prior_arr)
    log_probability_density = norm.logpdf(empirical_val, mean, std_dev)
    return log_probability_density



def _sample_prior_dist_worker(G,
                              ruggedness_fn,
                              sample_size_range,
                              sample_size,
                              _fluctuation_scale,
                              local):
    '''
    
    '''
    G_cpy = G.copy()
    
    if sample_size is not None:
        sampled_size = sample_size
    else:
        sampled_size = np.random.uniform(sample_size_range[0], sample_size_range[1])
    
    G_sampled = sample_values(G=G_cpy, sample_size=sampled_size)
    
    G_diffused = brownian_diffusion(G=G_sampled, fluctuation_scale=_fluctuation_scale)

    if not local:
        return ruggedness_fn(G_diffused)
    else:
        graph_ruggedness_de.compute_local_dirichlet_energy(G=G_diffused)
        local_dir = {node: G_diffused.nodes[node]['local_dirichlet'] for node in G_diffused.nodes()}
        return local_dir

def sample_prior_dist(G, 
                      ruggedness_fn, 
                      sample_size_range: tuple = (0.2, 0.6),
                      sample_size: float = None,
                      replicates=100, 
                      _fluctuation_scale=0,
                      local: bool =False,):
    '''
    
    '''

    results = Parallel(n_jobs=-1)(
        delayed(_sample_prior_dist_worker)(
            G, ruggedness_fn, sample_size_range, sample_size, _fluctuation_scale, local
        ) for _ in range(replicates)
    )

    if not local:
        prior_arr = np.array(results)
        return prior_arr[~np.isnan(prior_arr)]
    else:
        local_dir_dict = {}
        for local_dir in results:
            for node, value in local_dir.items():
                local_dir_dict.setdefault(node, []).append(value)
        return local_dir_dict


def sample_rugged_prior_dist(G, 
                      ruggedness_fn, 
                      replicates=100, 
                      local: bool =False):

    '''
    
    '''

    prior_dist = []
    local_dir_dict = {}
    for _, _ in enumerate(range(replicates)):

        G_cpy = G.copy()
        
        G_sampled = random_sample(G=G_cpy)

        if not local:
            prior_dist.append(ruggedness_fn(G_sampled))
        
        else:
            graph_ruggedness_de.compute_local_dirichlet_energy(G=G_sampled)
            for node in G_sampled.nodes():
                if node in local_dir_dict.keys():
                    local_dir_dict[node] += [G_sampled.nodes[node]['local_dirichlet']]
                else:
                    local_dir_dict[node] = [G_sampled.nodes[node]['local_dirichlet']]

    
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
        raise Exception('The graph must be connected.')

    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    values = np.array([G.nodes[node]['value'] for node in nodes])

    # Build adjacency matrix using nx.adjacency_matrix
    adjacency_matrix = nx.adjacency_matrix(G, nodelist=nodes, weight=None)#, format='csr')

    iterations = n_nodes ** 2

    for _ in range(iterations):

        if not np.any(np.isnan(values)):
            break

        values_nonan = np.nan_to_num(values, nan=0.0)
        valid_mask = (~np.isnan(values)).astype(float)

        neighbor_sums = adjacency_matrix.dot(values_nonan)
        neighbor_counts = adjacency_matrix.dot(valid_mask)

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_neighbor_signal = np.divide(
                neighbor_sums, neighbor_counts, out=np.zeros_like(neighbor_sums), where=neighbor_counts != 0
            )

        fluctuation = np.random.randn(n_nodes) * fluctuation_scale

        # Update values
        is_nan = np.isnan(values)
        values[is_nan] = avg_neighbor_signal[is_nan] + fluctuation[is_nan]
        values[~is_nan] = (values[~is_nan] + avg_neighbor_signal[~is_nan]) / 2 + fluctuation[~is_nan]

    # Update node values in the graph
    for idx, node in enumerate(nodes):
        G.nodes[node]['value'] = values[idx]

    return G

def random_sample(G):

    '''
    
    '''
    
    G = G.copy()

    if not nx.is_connected(G):
        raise Exception('The graph must be connected.')
    
    nodes = list(G.nodes())
    values = np.array([G.nodes[node]['value'] for node in nodes])
    mean_value = np.mean(values)
    std_value = np.std(values)

    new_values = np.random.normal(loc=mean_value, scale=std_value, size=len(nodes))

    # Update node values
    for idx, node in enumerate(nodes):
        G.nodes[node]['value'] = new_values[idx]
    
    return G

def sample_values(G: nx.Graph,
                  sample_size: float):
    '''
    
    '''
    
    G = G.copy()
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    num_sampled = int(sample_size * num_nodes)

    if num_sampled <= 2:
        raise ValueError('Sample size too small. Increase sample size.')

    values = np.array([G.nodes[node]['value'] for node in nodes])
    max_value = values.max()
    min_value = values.min()
    eligible_indices = np.where((values != max_value) & (values != min_value))[0]
    num_sampled = min(num_sampled - 2, len(eligible_indices))
    sampled_indices = np.random.choice(eligible_indices, size=num_sampled, replace=False)
    # Include max and min value nodes
    max_index = np.where(values == max_value)[0][0]
    min_index = np.where(values == min_value)[0][0]
    sampled_indices = np.concatenate(([max_index, min_index], sampled_indices))

    # Set non-sampled nodes to NaN
    sampled_mask = np.zeros(num_nodes, dtype=bool)
    sampled_mask[sampled_indices] = True
    values[~sampled_mask] = np.nan

    # Update node values
    for idx, node in enumerate(nodes):
        G.nodes[node]['value'] = values[idx]

    return G

def compute_marginal_likelihoods(h0_array: np.array,
                                 h1_array: np.array,
                                 empirical_val: float):
    '''
    
    '''
        
    # Fit distributions (use log probabilities to avoid underflow)
    mu_H0, std_H0 = stats.norm.fit(list(h0_array))
    mu_H1, std_H1 = stats.norm.fit(list(h1_array))

    # Compute log probabilities
    log_P_E_given_H0 = stats.norm.logpdf(empirical_val, loc=mu_H0, scale=std_H0)
    log_P_E_given_H1 = stats.norm.logpdf(empirical_val, loc=mu_H1, scale=std_H1)

    return log_P_E_given_H1 - log_P_E_given_H0