from scipy.linalg import eigh
import numpy as np 
import networkx as nx
from graph_ruggedness_de import compute_laplacian

def graph_fourier_transform(G: nx.Graph,
                            edge_weights: bool = False):
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
    norm_scalar = np.sum(abs_gft_coefficients)
    norm_gft_coefficients = abs_gft_coefficients / norm_scalar
    
    return norm_gft_coefficients