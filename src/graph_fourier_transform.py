from scipy.linalg import eigh
import numpy as np 
import networkx as nx
from graph_ruggedness_de import compute_laplacian

def graph_fourier_transform(G: nx.Graph,
                            edge_weights: bool = False,
                            return_norm: bool = True):
    '''
    Function to perform graph Fourier transform on a OHE networkX graph
    with fitness values stores as "value" node attributes. The GFT
    beta coefficients (i.e magnitudes) are returned as absolute values
    that have been normalised (i.e. sum to 1.0).

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
    
    return_norm: bool, default=`True`
        Boolean of whether to return the normalised coefficients.
    
    Returns:
    -------
    norm_gft_coefficients: np.array
        Numpy array of graph Fourier transform beta coefficients (i.e.
        magnitudes) as absolute values that have been normalised to 
        sum to 1.0. 
    '''
    
    laplacian = compute_laplacian(G=G,
                                  edge_weights=edge_weights)
    eigenvalues, eigenvectors = eigh(laplacian)
    signal = np.array([G.nodes[node]['value'] for node in G.nodes()])
    gft_coefficients = eigenvectors.T @ signal
    abs_gft_coefficients = np.abs(gft_coefficients)
    if not return_norm:

        return abs_gft_coefficients
    
    else:
        norm_scalar = np.sum(abs_gft_coefficients)
        norm_gft_coefficients = abs_gft_coefficients / norm_scalar
        
        return norm_gft_coefficients

def compute_hamming_weight(seq: str) -> float:
    '''
    Helper function to compute the Hamming weight of a binary sequence.
    Use is intended exclusively for OHE genotypes and not graphs where
    nodes encode amino acid sequences as `sequence` or the node name.

    Arguments:
    ----------
    seq: str
        The OHE genotype of a node.
    
    Returns:
    --------
    float
        The Hamming weight of the node. 
    '''

    return sum(int(bit) for bit in seq)

def categorize_eigenvectors(G: nx.Graph,
                            gft_coeff: np.array,
                            epsilon: float) -> dict:
    '''
Function to categorize GFT coefficients by their associated Hamming weights.

    Arguments:
    ----------
    G: nx.Graph
        The network graph.
    
    gft_coeff: np.array
        The GFT coefficients.
    
    eigenvectors: np.ndarray
        The matrix of eigenvectors of the graph Laplacian.

    epsilon: float
        Value for separating eigenvectors into orders of hamming 
        support.
    
    Returns:
    --------
    categorized_coeffs: dict
        Dictionary with keys 'first_order', 'second_order', 'higher_order' 
        and values being the sum of the squared GFT coefficients for each category.
    '''

    laplacian = compute_laplacian(G=G,
                                  edge_weights=False)
    
    eigenvalues, eigenvectors = eigh(laplacian)

    node_sequences = list(G.nodes())
    hamming_weights = {seq: compute_hamming_weight(seq) for seq in node_sequences}

    categorized_coeffs = {'first_order': 0.0, 'second_order': 0.0, 'higher_order': 0.0}

    threshold = epsilon 
    for idx, gft_coefficient in enumerate(gft_coeff):
        eigenvector = eigenvectors[:, idx]
        hamming_weight_support = {hamming_weights[seq] for seq, val in zip(node_sequences, eigenvector) if abs(val) > threshold}
        print(f"Eigenvector {idx}: Hamming weight support: {hamming_weight_support}")
        
        if hamming_weight_support == {1}:
            categorized_coeffs['first_order'] += abs(gft_coefficient) ** 2
        elif hamming_weight_support == {2}:
            categorized_coeffs['second_order'] += abs(gft_coefficient) ** 2
        else:
            categorized_coeffs['higher_order'] += abs(gft_coefficient) ** 2

    # Normalize the categorized coefficients
    total_energy = sum(categorized_coeffs.values())
    for key in categorized_coeffs:
        categorized_coeffs[key] /= total_energy

    return categorized_coeffs