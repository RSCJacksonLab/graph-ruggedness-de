import numpy as np
import networkx as nx
from scipy.optimize import minimize
from tqdm import tqdm
import random
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats
import math
import annoy
import concurrent.futures
import os
from sklearn.preprocessing import LabelEncoder


def sample_graph(G: nx.Graph,
                 sample_size: float) -> tuple:
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

    # Gather values and determine max/min fitness values
    values = np.array([G.nodes[node]['value'] for node in G])
    max_value = values.max()
    min_value = values.min()
    
    # Get the list of nodes, excluding max and min value nodes
    eligible_nodes = [node for node in G if G.nodes[node]['value'] not in (max_value, min_value)]
    
    # Ensure we don't sample more than needed
    if len(eligible_nodes) > (num_sampled - 2):
        sampled_nodes = np.random.choice(eligible_nodes, size=num_sampled - 2, replace=False).tolist()
    else:
        sampled_nodes = eligible_nodes

    # Add exactly one node with max and one with min value
    sampled_nodes.append(next(node for node in G if G.nodes[node]['value'] == max_value))
    sampled_nodes.append(next(node for node in G if G.nodes[node]['value'] == min_value))
    
    # Build the subgraph and get sampled values
    G_sampled = G.subgraph(sampled_nodes).copy()
    sampled_values = [G_sampled.nodes[node]['value'] for node in sampled_nodes]

    return G_sampled, sampled_nodes, sampled_values

def add_ohe_knn_edges(G: nx.Graph,
                      k: int,
                      verbose: bool = True) -> None:
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
    ohe_array = np.array([G.nodes[seq]['ohe'] for seq in sequences])
    print('Computing all vs. all OHE distance matrix.')
    dist_mat = distance_matrix(ohe_array, ohe_array)
    print('Done.')

    print('Finding K-nearest neighbors.')
    indices = np.argsort(dist_mat, axis=1)[:, 1:k+1]
    distances = np.sort(dist_mat, axis=1)[:, 1:k+1]
    print('Done.')

    # Prepare edge data
    idx_rows = np.repeat(np.arange(len(sequences)), k)
    idx_cols = indices.flatten()
    distances_flat = distances.flatten()
    inv_weights = 1 / distances_flat

    # Filter out self-loops and duplicate edges
    mask = idx_rows != idx_cols
    edges = [(sequences[i], sequences[j], {'distance': d, 'inv_weight': w})
             for i, j, d, w in zip(idx_rows[mask], idx_cols[mask], distances_flat[mask], inv_weights[mask])
             if not G.has_edge(sequences[i], sequences[j])]

    G.add_edges_from(edges)
    
    if verbose: 
        print(f'Added {len(edges)} KNN edges.')

    # Ensure full connectivity
    ensure_full_connectivity(G)    

def add_ohe_knn_edges_approx(G: nx.Graph,
                             k: int,
                             verbose : bool = True) -> None:
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
    if verbose:
        print('Building approx. NN index.')

    t = annoy.AnnoyIndex(f, 'hamming')
    
    if verbose:
        print('Done.')

    for i, vec in tqdm(enumerate(ohe_list), desc='Adding OHE vectors to index.'):
        t.add_item(i, vec)

    if verbose:
        print('Building approx. NN search tree(s).')

    t.build(2) 
    
    if verbose:
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
    
    # Add edges between disconnected components to ensure full connectivity
    ensure_full_connectivity(G=G)

def ensure_full_connectivity(G: nx.Graph,
                             verbose: bool = True) -> None:
    '''
    Function to ensure that a graph is fully connected by adding edges
    between disconnected components based on the closest pair of nodes
    between components.

    Arguments:
    ----------
    G: networkX.Graph
        The graph to ensure full connectivity.
    '''
    if not nx.is_connected(G):
        
        if verbose:
            print('Connecting components')

        components = list(nx.connected_components(G))
        
        # Sort components by size (optional, to connect smaller components first)
        components.sort(key=len, reverse=True)

        # Iteratively connect all components
        for i in range(1, len(components)):
            component_a = components[i-1]
            component_b = components[i]

            min_distance = float('inf')
            best_pair = None

            # Find the closest pair of nodes between component_a and component_b
            for node_a in component_a:
                for node_b in component_b:
                    # Calculate the distance between node_a and node_b
                    distance = np.linalg.norm(G.nodes[node_a]['ohe'] - G.nodes[node_b]['ohe'])
                    if distance < min_distance:
                        min_distance = distance
                        best_pair = (node_a, node_b)
            
            # Add the edge between the closest pair of nodes
            if best_pair:
                G.add_edge(best_pair[0],
                           best_pair[1],
                           distance=min_distance,
                           inv_weight=1/min_distance)

def add_hamming_edges(G,
                      threshold: int = 1,
                      verbose: bool = True) -> None:
    '''
    Inplace function to compute hamming neighbor edges to a OHE graph.
    Hamming edges are computed using scipy.spatial.distance_matrix,
    which is exhaustive. Therefore, hamming edges are exact. 

    Arguments:
    ----------
    G: networkX.Graph
        A networkX graph of OHE sequences as nodes.  

    threshold: int, default=`1`
        The mutation threshold to add an edge.
    '''

    sequences = list(G.nodes())
    seq_indices = np.array([G.nodes[seq]['sequence'] for seq in sequences])
    label_encoder = LabelEncoder()
    label_encoder.fit([aa for seq in sequences for aa in seq])
    seq_indices = np.array([label_encoder.transform(list(seq)) for seq in sequences])

    # Compute pairwise Hamming distances
    distances = pdist(seq_indices, metric='hamming') * seq_indices.shape[1]
    dist_mat = squareform(distances)

    # Threshold for Hamming distance

    # Get indices where distance <= threshold
    i_upper, j_upper = np.triu_indices_from(dist_mat, k=1)
    mask = dist_mat[i_upper, j_upper] <= threshold + 1e-9
    i_edges = i_upper[mask]
    j_edges = j_upper[mask]

    # Add edges
    edges_to_add = [(sequences[i], sequences[j]) for i, j in zip(i_edges, j_edges)]
    G.add_edges_from(edges_to_add)
    
    if verbose:
        print(f'Added {len(edges_to_add)} Hamming edges.')

def build_ohe_graph(seq_ls,
                    values, edges: bool = True,
                    hamming_edges: bool = False,
                    approximate: bool = False) -> nx.Graph:
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

    # Initialize graph
    G = nx.Graph()
    
    # Flatten the list of sequences to a list of amino acids
    all_aa = [aa for seq in seq_ls for aa in seq]
    
    # Encode amino acids as integers
    label_encoder = LabelEncoder()
    label_encoder.fit(all_aa)
    
    # Transform sequences into integer indices
    seq_indices = np.array([label_encoder.transform(list(seq)) for seq in seq_ls])
    
    # One-hot encode sequences
    num_aa = len(label_encoder.classes_)
    ohe_array = np.eye(num_aa)[seq_indices]  # Shape: (num_sequences, seq_length, num_aa)
    ohe_array = ohe_array.reshape(len(seq_ls), -1)  # Flatten to (num_sequences, seq_length * num_aa)
    
    # Add nodes to the graph
    for idx, seq in enumerate(seq_ls):
        G.add_node(seq, sequence=seq, ohe=ohe_array[idx], value=values[idx])
    
    # Add edges as before
    if edges:
        if hamming_edges:
            if approximate:
                raise NotImplementedError('Approximate hamming edges implementation broken.')
            else:
                add_hamming_edges(G)
        else:
            k = int(np.sqrt(len(G)))
            if approximate:
                add_ohe_knn_edges_approx(G, k=k)
            else:
                add_ohe_knn_edges(G, k=k)
    
    return G

def compute_local_dirichlet_energy(G: nx.Graph,
                                   approximate: bool = False) -> None:

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
    nodes = list(G.nodes())
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    values = np.array([G.nodes[node]['value'] for node in nodes])

    # Build adjacency matrix using nx.adjacency_matrix
    adj_matrix = nx.adjacency_matrix(G, nodelist=nodes, weight=None)
    adj_matrix = adj_matrix + adj_matrix.T  # Ensure symmetry
    adj_matrix[adj_matrix > 1] = 1  # Remove multiple edges

    for idx, node in tqdm(enumerate(nodes), desc='Computing local Dirichlet energy.'):
        # Use getrow to retrieve the row as a 2D sparse matrix
        neighbor_indices = adj_matrix.getrow(idx).nonzero()[1]
        sub_indices = np.append(neighbor_indices, idx)
        sub_values = values[sub_indices]
        sub_adj = adj_matrix[sub_indices][:, sub_indices]

        if approximate:
            diffs = sub_values[:, None] - sub_values[None, :]
            sq_diffs = diffs ** 2
            local_dirichlet = np.sum(sq_diffs * sub_adj.toarray()) / 2
        else:
            # Corrected line: use .flatten() instead of .A1
            degree_vector = np.array(sub_adj.sum(axis=1)).flatten()
            degree_matrix = np.diag(degree_vector)
            laplacian = degree_matrix - sub_adj.toarray()
            local_dirichlet = sub_values @ laplacian @ sub_values

        G.nodes[node]['local_dirichlet'] = local_dirichlet

def compute_dirichlet_energy(G: nx.Graph,
                             edge_weights: bool = False) -> float:

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
        laplacian = nx.laplacian_matrix(G, weight=None).toarray()
    else:
        laplacian = nx.laplacian_matrix(G, weight='inv_weight').toarray()
    dir_en = y @ laplacian @ y
    return dir_en

def compute_dirichlet_energy_approximate(G: nx.Graph, 
                                         edge_weights: bool = False) -> float:
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

    nodes = list(G.nodes())
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    values = np.array([G.nodes[node]['value'] for node in nodes])

    # Get edge indices
    edges = np.array([(node_indices[u], node_indices[v]) for u, v in G.edges()])
    diffs = values[edges[:, 0]] - values[edges[:, 1]]
    sq_diffs = diffs ** 2

    if edge_weights:
        weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
        dir_en = np.sum(sq_diffs * weights)
    else:
        dir_en = np.sum(sq_diffs)
    return dir_en

def compute_pairwise_difference(edge: tuple,
                                G: nx.Graph,
                                edge_weights: bool = False) -> float:
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

def compute_elementary_landscape(G: nx.Graph,
                                 n: int,
                                 edge_weights: bool = False) -> nx.Graph:
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
        laplacian = nx.laplacian_matrix(G_elementary)
    
    elif edge_weights:
        laplacian = laplacian = nx.laplacian_matrix(G_elementary)
        
    laplacian = laplacian.toarray()
    
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    nth_eigenvector = eigenvectors[:, n]

    for i, node in enumerate(G_elementary.nodes()):
        G_elementary.nodes[node]['value'] = nth_eigenvector[i]
    
    return G_elementary