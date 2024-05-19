import numpy as np
import networkx as nx
from scipy.optimize import minimize
from tqdm import tqdm
import random
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
import math
import annoy
import concurrent.futures
import os

def sample_graph(G: nx.Graph,
                 sample_size: float):
    '''
    Function to subsample a networkX graph according to a sampling 
    proportion. 
    
    Arguments:
    ----------
    G: networkX.Graph
        A KNN or hamming graph as a networkX graph object. 

    sample_size: float
        Proportion of the networkX graph to subsample. 
    
    Returns:
    --------
    G_sampled: networkX.Graph
        Graph of sample_size * len(G) nodes sampled at random from G.
    
    sampled_nodes: list
        List of nodes sampled from G in G_sampled.
    
    sampled_values: list
        List of values sampled from G in G_sampled. 
    '''

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
    sampled_nodes.extend([node for node in G if G.nodes[node]['value'] in (max_value, min_value)])
    G_sampled = G.subgraph(sampled_nodes).copy()
    sampled_values = [G_sampled.nodes[node]['value'] for node in sampled_nodes]

    return G_sampled, sampled_nodes, sampled_values


def sample_graph_degree(G: nx.Graph,
                       degree: int,
                       ref: str = None, 
                       compute_de: bool = False,
                       approximate: bool = False):
    '''
    Function to subsample a networkX graph according to a degree value
    from a reference node. 
    
    Arguments:
    ----------
    G: networkX.Graph
        A KNN or hamming graph as a networkX graph object. 

    degree: int
        The degree of edges to sample around the reference node. 
    
    ref: str
        The node name for the reference point. If None, the reference
        node is randomly sampled from all nodes in the graph. Default
        is None (i.e. random node selection).
    
    compute_de: bool
        Boolean for whether to compute the Dirichlet Energy at each 
        level of the graph. Default value is False. 
    
    approximate: bool
        Boolean for whether to use approximate Dirichlet energy, if
        comppte_de. Default value is False. 
    
    Returns:
    --------
    G_sampled: networkX.Graph
        Subgraph of G that spans all nodes that are incident to degree
        with the reference node. 
    '''
    
    if ref:
        ref_node = G.nodes[ref]
    else:
        ref_node = np.random.choice(list(G.nodes()))

    G_local = nx.Graph()
    next_level = set([ref_node])
    degree_dict = {}
    visited = set()

    for i in range(degree):

        current_level = list(next_level)
        next_level = set()
        
        for _, node in tqdm(enumerate(current_level),
                            desc='Enumerating nodes in current level.',
                            total=len(current_level)):
            
            neighbors = list(G.neighbors(node))
            G_local.add_edges_from([(node, neighbor) for neighbor in neighbors])
            visited.add(node)
            next_level.update([neighbor for neighbor in neighbors if neighbor not in visited])

        for node in G_local.nodes():
            G_local.nodes[node]['value'] = G.nodes[node].get('value', None)
        
        if compute_de:
            scaling_factor = np.sqrt(G.number_of_nodes()) / np.sqrt(G_local.number_of_nodes())
            if not approximate:
                de = compute_dirichlet_energy(G=G_local) / G_local.number_of_nodes() * scaling_factor
            elif approximate:
                de = compute_dirichlet_energy_approximate(G=G_local) / G_local.number_of_nodes() * scaling_factor
            degree_dict[i+1] = de
    
    if compute_dirichlet_energy:
        return G_local, ref_node, degree_dict
    else:
        return G_local, ref_node

def count_seq_diff(ref_node: str,
                   node_ls: list):
    '''
    Function to count the number of differences between a reference
    node sequence and a list of node sequences. Requires that all node
    values / sequences are the same length (i.e. the graph is made from
    aligned sequences).

    Arguments:
    ----------
    ref_node: str
        Value of a reference node to compare the node list against. 
    
    node_ls: list
        List of node sequences to be compared against the reference
        node sequence. 
    
    Returns:
    --------
    diff_ls: list[int]
        List of with length of node_ls with the number of sequence 
        differences between the indexed node value and the reference
        node value / sequence. 
    '''

    node_array = np.array([list(s) for s in node_ls])
    ref_array = np.array(list(ref_node))
    diff = np.not_equal(node_array, ref_array)
    diff_ls = np.sum(diff, axis=1).tolist()
    
    return diff_ls

def add_ohe_knn_edges(G: nx.Graph,
                      k: int):
    '''
    Inplace function to compute K-nearest neighbor edges to a OHE
    graph. KNN edges are computed using scipy.spatial.distance_matrix,
    which is exhaustive. Therefore, KNN edges are exact. 

    Arguments:
    ----------
    G: networkX.Graph
        A networkX graph of OHE sequences as nodes. 
    
    k: int
        The number of nearest neighbors to add edges to. 
    '''

    sequences = list(G.nodes())
    ohe_list = [G.nodes[seq]['ohe'] for seq in sequences]
    print('Computing all vs. all OHE distance matrix.')
    dist_mat = distance_matrix(ohe_list, ohe_list)
    print('Done.')

    nn_model = NearestNeighbors(n_neighbors=k, metric='precomputed')
    print('Fitting all vs. all OHE distance model.')
    nn_model.fit(dist_mat)
    print('Done.')

    distances, indices = nn_model.kneighbors(dist_mat)

    for idx, neighbors in tqdm(enumerate(indices), desc='Adding KNN edges.'):
            current_node = sequences[idx]
            for neighbor_idx, neighbor_node in enumerate(neighbors):
                if current_node != sequences[neighbor_node] and not G.has_edge(current_node, sequences[neighbor_node]):
                    distance = distances[idx][neighbor_idx]
                    G.add_edge(current_node, sequences[neighbor_node], distance=distance, inv_weight=1/(distance))

def add_ohe_knn_edges_approx(G: nx.Graph,
                             k: int):
    '''
    Inplace function to compute K-nearest neighbor edges to a OHE
    graph. KNN edges are annoy approximate package, which is an
    implementation of tree-search. Therefore, results are approximate
    and accuracy can be improved by increasing rthe number of trees
    in the forest. 

    Arguments:
    ----------
    G: networkX.Graph
        A networkX graph of OHE sequences as nodes. 
    
    k: int
        The number of nearest neighbors to add edges to. 
    '''
    sequences = list(G.nodes())
    ohe_list = [G.nodes[seq]['ohe'] for seq in sequences]
    
    f = ohe_list[0].shape[0]
    print('Building approx. NN index.')
    t = annoy.AnnoyIndex(f, 'hamming')
    print('Done.')

    for i, vec in tqdm(enumerate(ohe_list), desc='Adding OHE vectors to index.'):
        t.add_item(i, vec)

    print('Building approx. NN search tree(s).')
    t.build(2) #TODO: test trade-off in accuracy speed for more trees in forest. 
    print('Done.')
    
    for _, i in tqdm(enumerate(range(len(ohe_list))), desc='Adding approximate KNN edges.'):
    # Fetch the 'n' nearest neighbors for the item
        nn_indices, distances = t.get_nns_by_item(i,
                                                  k,
                                                  include_distances=True,
                                                  search_k=-1)
        absolute_distances = [distance * f for distance in distances]
        
        for j, distance in zip(nn_indices, absolute_distances):
            if i != j:
                if not G.has_edge(sequences[i], sequences[j]):
                    G.add_edge(sequences[i], sequences[j], distance=distance, inv_weight=1/(distance))

def add_hamming_edges(G: nx.Graph):
    '''
    Inplace function to compute hamming neighbor edges to a OHE graph.
    Hamming edges are computed using scipy.spatial.distance_matrix,
    which is exhaustive. Therefore, hamming edges are exact. 

    Arguments:
    ----------
    G: networkX.Graph
        A networkX graph of OHE sequences as nodes.  
    '''
    sequences = list(G.nodes())
    ohe_list = [G.nodes[seq]['ohe'] for seq in sequences]
    
    print('Computing all vs. all OHE distance matrix.')
    dist_mat = squareform(pdist(ohe_list, 'hamming')) * ohe_list.shape[1]    
    print('Done.')

    # Since a single amino acid change in one-hot encoding would result in 
    # a Euclidean distance of sqrt(2).
    threshold = np.sqrt(2)
    
    for i, row in tqdm(enumerate(dist_mat), desc='Adding hamming edges.'):
        current_node = sequences[i]
        for j, distance in enumerate(row):
            if i != j and distance <= threshold:
                # This condition now properly checks for the "Hamming distance" of 1 
                # in terms of single amino acid changes in OHE sequences.
                # Adjusted to allow for slight numerical inaccuracies.
                if not G.has_edge(current_node, sequences[j]):
                    G.add_edge(current_node, sequences[j])

def add_hamming_edges_approx(G: nx.Graph,
                             n: int):
    '''
    Inplace function to compute hamming neighbor edges to a OHE graph.
    Hamming edges are computed using annoy, which is approximate. Trade
    off between accuracy and speed can be improved by changing the
    number of trees in the index. 

    Arguments:
    ----------
    G: networkX.Graph
        A networkX graph of OHE sequences as nodes. 
    
    n: int
        The number of edges to add between each node. Assumes that the
        OHE dataset in the graph is exhaustive and combinatorially
        complete. 
    '''
    sequences = list(G.nodes())
    ohe_list = [G.nodes[seq]['ohe'] for seq in sequences]

    #TODO: Threshold distance is not correct - no edges in graph are produced. 
    f = len(ohe_list[0])  # Length of the one-hot encoded vectors
    threshold_distance = 1.0 / f  # Calculating threshold for a Hamming distance of 1
    
    f = ohe_list[0].shape[0]
    print('Building approx. NN index.')
    t = annoy.AnnoyIndex(f, 'hamming')
    print('Done.')

    for i, vec in tqdm(enumerate(ohe_list), desc='Adding OHE vectors to index.'):
        t.add_item(i, vec)

    print('Building approx. NN search tree(s).')
    t.build(2) #TODO: test trade-off in accuracy speed for more trees in forest. 
    print('Done.')
    
    for _, i in tqdm(enumerate(range(len(ohe_list))), desc='Adding approximate hamming edges.'):
    # Fetch the 'n' nearest neighbors for the item
        nn_indices, distances = t.get_nns_by_item(i,
                                                  n,
                                                  include_distances=True,
                                                  search_k=-1)
        for idx, j in enumerate(nn_indices):
            if i != j and distances[idx] <= threshold_distance:
                if not G.has_edge(sequences[i], sequences[j]):
                    G.add_edge(sequences[i], sequences[j])


def build_ohe_graph(seq_ls: list,
                    values: list,
                    edges: bool = True,
                    hamming_edges: bool = False,
                    approximate: bool = False,
                    n: int = None):
    '''
    Function to build OHE graph with either hamming or KNN edges. 

    Arguments:
    ----------
    seq_ls: list
        List of sequences (as amino acid tokens or sparse OHE
        representation).
    
    values: list
        List of fitness values (signal) matched by index to the seq_ls
        index. 
    
    edges: bool
        Boolean of whether to include computaton of edges in OHE graph
        construction. Default is True. Must be False if sampling the
        graph to enable re-determination of the graph topology with
        KNN methods. 
    
    hamming_edges: bool
        Boolean of whether edges should be computed as hamming (True)
        of KNN (False). Default is False. 
    
    approximate: bool
        Boolean of whether to use approximate method to compute edges
        over the graph. 
    
    n: None, int
        Number of edges to compute between nodes if hamming_edges and
        approximate are both True. Required by add_hamming_edges_approx
        function. 
    '''
    if approximate and hamming_edges:
        assert type(n) == int

    G = nx.Graph()

    aa_ls = set(aa for seq in seq_ls for aa in seq)
    ohe_dict = {aa: np.eye(len(aa_ls))[i] for i, aa in tqdm(enumerate(aa_ls), desc='Computing OHE dictionary.')}

    for idx, seq in tqdm(enumerate(seq_ls), desc='Adding nodes to graph.'):
        ohe_seq = np.array([ohe_dict[aa] for aa in seq]).flatten()
        G.add_node(seq, sequence=seq, ohe=ohe_seq, value=values[idx])
    
    if edges:
        if hamming_edges:
            if approximate:
                add_hamming_edges_approx(G=G,
                                        n=n)
            else:
                add_hamming_edges(G=G)

        else:
            k = int(np.sqrt(len(G)))
            if approximate:
                add_ohe_knn_edges_approx(G=G,
                                        k=k)
            else:
                add_ohe_knn_edges(G=G, k=k)
    
    return G


def compute_laplacian(G: nx.Graph,
                      edge_weights: bool = False):

    '''
    Function to compute the Laplacian matrix of a OHE network graph.
    Laplacian is computed as the degree_matrix - the adjacency matrix.

    Arguments:
    ----------
    G: networkX.Graph
        OHE networkX graph with edges. 

    edge_weights: bool
        Boolean of whether edges are weighted in the network graph. 
    
    Returns:
    --------
    laplacian: np.array
        Laplacian matrix. 
    '''
    if not edge_weights:
        adj_matrix_sparse = nx.adjacency_matrix(G)    
    
    if edge_weights:
        adj_matrix_sparse = nx.adjacency_matrix(G, weight='inv_weight')    

    adj_matrix = adj_matrix_sparse.toarray()


    # Ensure the adjacency matrix is symmetric.
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

    # Compute the Laplacian matrix.
    diag_mat = np.diag(np.sum(adj_matrix, axis=0))
    laplacian = diag_mat - adj_matrix

    return laplacian

def compute_local_dirichlet_energy(G: nx.Graph,
                                   approximate: bool = False):

    '''
    Inplace function to compute the nodewise local Dirichlet energy. 
    Energies are returned as "local_dirichlet" node attributes.

    Arguments:
    ----------
    G: nx.Graph
        NetworkX graph of OHE sequences with fitnesses stored as the
        "value" node attributes.
    Approximate: bool
        Whether local Dirichlet energies should be computed using the
        approximate method. Default is False
    '''

    for _, node in tqdm(enumerate(G.nodes()), desc='Computing local Dirichlet energy.'):
        neighbors = list(G.neighbors(node))
        subgraph_nodes = neighbors + [node]
        G_local = G.subgraph(subgraph_nodes).copy()
        if approximate:
            G.nodes[node]['local_dirichlet'] = compute_dirichlet_energy_approximate(G=G_local)
        else:
            G.nodes[node]['local_dirichlet'] = compute_dirichlet_energy(G=G_local)

def _compute_energy_for_nodes(G: nx.Graph,
                             nodes: list,
                             approximate: bool = False):
    
    '''
    Function to compute energy for a list of nodes. Run under the hood
    during parallel computation of local Dirichlet energies. Not
    intended for use outside of the parallel local Dirichlet energy
    computation. 

    Arguments:
    ----------
    G: nx.Graph
        Graph to compute local Dirichlet energies over.
    
    nodes: list
        List of nodes to make subgraph of. 

    approximate: bool
        Boolean of whether to use the approximate square edge
        different when computing the Dirichlet energy. 

    Returns:
    --------
    local_energies: dict
        Dictionary of local energies with nodes as keys and local
        energies as values. 
    '''
    local_energies = {}
    for node in nodes:
        neighbors = list(G.neighbors(node))
        subgraph_nodes = neighbors + [node]
        G_local = G.subgraph(subgraph_nodes).copy()
        if approximate:
            local_energies[node] = compute_dirichlet_energy_approximate(G=G_local)
        else:
            local_energies[node] = compute_dirichlet_energy(G=G_local)
    return local_energies

def compute_local_dirichlet_energy_parallel(G: nx.Graph,
                                            approximate: bool = False,
                                            num_cpus: int = os.cpu_count()):
    '''
    Inplace function to compute the nodewise local Dirichlet energy. 
    Energies are returned as "local_dirichlet" node attributes. Run
    over multiple CPUs. 

    Arguments:
    ----------
    G: nx.Graph
        NetworkX graph of OHE sequences with fitnesses stored as the
        "value" node attributes.
    Approximate: bool
        Whether local Dirichlet energies should be computed using the
        approximate method. Default is False
    num_cpus: int
        The number of CPUs to run the process over.
    '''
    
    nodes = list(G.nodes())
    chunk_size = len(nodes) // num_cpus + 1
    node_chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [executor.submit(_compute_energy_for_nodes, G, chunk, approximate) for chunk in node_chunks]
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc='Computing local Dirichlet energy in parallel.'):
            local_energies = future.result()
            for node, energy in local_energies.items():
                G.nodes[node]['local_dirichlet'] = energy

def compute_dirichlet_energy(G: nx.Graph,
                             edge_weights: bool = False):

    '''
    Function to compute the Dirichlet energy over a graph. The energy
    is computed as the y transpose dot Laplacian dot y.

    Arguments: 
    ----------
    G: networkX.Graph
        NetworkX graph of OHE sequences with either KNN or hamming 
        edges. The fitness signal must be stored as the "value"
        node attribute. 
    
    edge_weights: bool
        Boolean of weather to weight the adjacency matrix when 
        computing the Laplacian matrix. Edge weights must be stored
        as "weight" edge attribute. Default is False. 
    
    Returns:
    --------
    dir_en: np.array
        Numpy.array of Dirichlet energy with shape (1, ). 
    '''

    y = np.array([G.nodes[node]['value'] for node in G.nodes()])
    if not edge_weights:
        laplacian = compute_laplacian(G=G,
                                      edge_weights=False)
    elif edge_weights:
        laplacian = compute_laplacian(G=G,
                                edge_weights=True)

    dir_en = y.dot(laplacian).dot(y)

    # TODO: normalisation by average OHE edge length to make more interpretable. 
    return dir_en

def compute_dirichlet_energy_approximate(G: nx.Graph, 
                                         edge_weights: bool = False):
    '''
    Function to compute the Dirichlet energy over a graph. The energy
    is computed as summed square of differences in fitness over each 
    edge of the graph. 

    Arguments: 
    ----------
    G: networkX.Graph
        NetworkX graph of OHE sequences with either KNN or hamming 
        edges. The fitness signal must be stored as the "value"
        node attribute. 
    
    edge_weights: bool
        Boolean of weather to weight the adjacency matrix when 
        computing the Laplacian matrix. Edge weights must be stored
        as "weight" edge attribute. Default is False. 
    
    Returns:
    --------
    dir_en: np.array
        Numpy.array of Dirichlet energy with shape (1, ). 
    '''
    dir_en = sum((G.nodes[i]['value'] - G.nodes[j]['value'])**2 * (G[i][j]['weight'] if edge_weights else 1)
             for _,(i, j) in tqdm(enumerate(G.edges()), desc='Computing pairwise signal differences in Laplacian approximation.'))
    return dir_en

def compute_pairwise_difference(edge: tuple,
                                G: nx.Graph,
                                edge_weights: bool = False):
    '''
    Objective function of parallel, approximate dirichlet energy
    calculation. 
    
    Arguments:
    ----------
    edge: tuple
        The graph edge tuple. 
    
    G: nx.Graph
        The graph object
    
    edge_weights: bool
        Boolean of whether to include edge weights. Default value is
        False. 
    
    Returns:
    --------
    sqr_difference: float
        The squared difference in the `value` attribute over the edge.
    '''

    i, j = edge
    weight = G[i][j]['weight'] if edge_weights else 1
    sqr_difference = (G.nodes[i]['value'] - G.nodes[j]['value']) ** 2 * weight

    return sqr_difference

def compute_dirichlet_energy_approximate_paralel(G: nx.Graph,
                                                 edge_weights=False,
                                                 num_cpus=os.cpu_count()):
    '''
    Function to compute the Dirichlet energy over a graph. The energy
    is computed as summed square of differences in fitness over each 
    edge of the graph. Function is run over parallel processes. 

    Arguments:
    ----------
    G: networkX.Graph
        NetworkX graph of OHE sequences with either KNN or hamming 
        edges. The fitness signal must be stored as the "value"
        node attribute. 
    
    edge_weights: bool
        Boolean of weather to weight the adjacency matrix when 
        computing the Laplacian matrix. Edge weights must be stored
        as "weight" edge attribute. Default is False. 
    
    num_cpus: int
        The number of CPU threads to parrallelize processing over.
        Default value is the maximum available.
    
    Returns:
    --------
    dir_en: float
        Float of the Dirichlet energy with shape.
    '''
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
        
        futures = [executor.submit(compute_pairwise_difference, edge, edge_weights, G) for edge in G.edges()]
        dir_en = 0
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Computing pairwise signal differences in Laplacian approximation.'):
            dir_en += future.result()
    
    return dir_en

def compute_elementary_landscape(G: nx.Graph,
                                 n: int,
                                 edge_weights: bool = False):
    '''
    Function to compute the nth eigenvector of the Laplacian matrix
    from a OHE network graph with either hamming or KNN edges. 

    Arguments:
    ----------
    G: networkX.Graph
        NetworkX graph of OHE sequences with either KNN or hamming 
        edges. The fitness signal must be stored as the "value"
        node attribute. 
    
    n: int
        Index of the Laplacian eigenvector to compute. 
    
    edge_weights: bool
        Boolean of weather to weight the adjacency matrix when 
        computing the Laplacian matrix. Edge weights must be stored
        as "weight" edge attribute. Default is False. 
    
    Returns:
    --------
    G_elementary: networkX.Graph 
        NetworkX graph with identical topology to G. Laplacian 
        eigenvalues are stored as "value" node attributes. 
    '''

    G_elementary = G.copy()
    if not edge_weights:
        laplacian = compute_laplacian(G=G_elementary)
    
    elif edge_weights:
        laplacian = compute_laplacian(G=G_elementary,
                                      edge_weights=True)
        
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    nth_eigenvector = eigenvectors[:, n]

    for i, node in enumerate(G_elementary.nodes()):
        G_elementary.nodes[node]['value'] = nth_eigenvector[i]
    
    return G_elementary



##########################
# Brownian diffusion model
# Warning poor performance
##########################

def compute_log_probability(prior_arr, 
                            empirical_val):

    '''
    
    '''

    mean = np.mean(prior_arr)
    std_dev = np.std(prior_arr)
    probability_density = norm.pdf(empirical_val, mean, std_dev)
    return np.log(probability_density)


def parameterise_sample_size(G,
                             min_sample_size, 
                             max_sample_size, 
                             step):
    
    '''
    
    '''
    
    probabilities = []
    for sample_size in np.arange(min_sample_size, max_sample_size, step):
        G_n_eigh = compute_elementary_landscape(G=G, n=5)
        dir_eigh = compute_dirichlet_energy(G=G_n_eigh)

        prior_dist = sample_prior_dist(G, 
                                       sample_size=sample_size, 
                                       ruggedness_fn=compute_dirichlet_energy,
                                       replicates=10)
        mean = np.mean(prior_dist)
        std_dev = np.std(prior_dist)
        probabilities.append(norm.pdf(dir_eigh, mean, std_dev))

    max_val = max(x for x in probabilities if not math.isnan(x))
    optimal_idx = probabilities.index(max_val)
    
    return np.arange(min_sample_size, max_sample_size, step)[optimal_idx], probabilities[optimal_idx]

def sample_prior_dist(G, 
                      sample_size, 
                      ruggedness_fn, 
                      replicates=100, 
                      fluctuation_scale=0,
                      local=False):

    '''
    
    '''

    prior_dist = []
    local_dir_dict = {}
    for _, _ in enumerate(range(replicates)):

        G_cpy = G.copy()
        G_sampled, _, _ = sample_graph(G=G_cpy, sample_size=sample_size)
        brownian_diffusion(G=G_sampled, fluctuation_scale=fluctuation_scale)
        if not local:
            prior_dist.append(ruggedness_fn(G_sampled))
        else:
            compute_local_dirichlet_energy(G=G_sampled)
            for node in G_sampled.nodes():
                if node in local_dir_dict.keys():
                    local_dir_dict[node] += [G_sampled.nodes[node]['local_dirichlet']]
                else:
                    local_dir_dict[node] = [G_sampled.nodes[node]['local_dirichlet']]

    
    if not local:
        return np.array(prior_dist)
    else:
        return local_dir_dict
    
def brownian_diffusion(G, fluctuation_scale):

    '''
    
    '''

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
