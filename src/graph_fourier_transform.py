import numpy as np 
import networkx as nx

def graph_fourier_transform(G: nx.Graph,
                            edge_weights: bool = False,
                            return_norm: bool = True,
                            absolute_val: bool = True):
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

    absolute_val: bool, default=`True`
        Boolean of whether to return the absolute values of the fourier
        coefficients.
    
    Returns:
    -------
    norm_gft_coefficients: np.array
        Numpy array of graph Fourier transform beta coefficients (i.e.
        magnitudes) as absolute values that have been normalised to 
        sum to 1.0. 
    '''
    
    laplacian = nx.laplacian_matrix(G)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
    signal = np.array([G.nodes[node]['value'] for node in G.nodes()])
    gft_coefficients = eigenvectors.T @ signal

    if absolute_val:

        gft_coefficients = np.abs(gft_coefficients)
    
    if not return_norm:

        return gft_coefficients
    
    else:
        gft_coefficients = np.abs(gft_coefficients)
        norm_scalar = np.sum(gft_coefficients)
        norm_gft_coefficients = gft_coefficients / norm_scalar
        
        return norm_gft_coefficients

