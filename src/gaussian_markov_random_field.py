import numpy as np
import networkx as nx

def compute_laplacian_spectrum(G: nx.Graph) -> tuple:
    """
    Computes the eigenvalues and eigenvectors of the Laplacian matrix
    of graph G.
    
    Arguments:
    -----------
    G : networkx.Graph
        The input graph.

    Returns:
    --------
    laplacian : scipy.sparse matrix
        The Laplacian matrix of the graph.

    eigenvalues : np.ndarray
        The computed eigenvalues.

    eigenvectors : np.ndarray
        The computed eigenvectors.
    """
    
    laplacian = nx.normalized_laplacian_matrix(G, weight='inv_weight').asfptype()
    L_dense = laplacian.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
    
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return laplacian, eigenvalues, eigenvectors

def precompute_GMRF_stats(G: nx.Graph) -> tuple:
    """
    Function to precompute GMRF spectral and statistical quantities.

    Arguments:
    ----------
    G : nx.Graph
        The fitness landscape graph. 
    
    Returns:
    --------
    f_hat : np.array
        The Fourier transformed graph signal. 
    
    eigenvalues : np.array  
        The Laplacian eigenvalues. 
    
    sigma_squared : float
        The empirical variance in the signal. 
    """
    laplacian = nx.normalized_laplacian_matrix(G, weight='inv_weight').asfptype()
    L_dense = laplacian.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
    # Sort by ascending eigenvalue:
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    signal = np.array([G.nodes[node]['value'] for node in G.nodes()])
    mu = np.mean(signal)
    signal_centered = signal - mu
    f_hat = eigenvectors.T @ signal_centered
    sigma_squared = np.var(signal_centered, ddof=1)
    return f_hat, eigenvalues, sigma_squared


def compute_log_likelihood_H0(f_hat: np.ndarray,
                              eigenvalues: np.ndarray,
                              t: float,
                              sigma_squared: float,
                              epsilon: float = 1e-8,
                              ) -> tuple:
    """
    Function to compute the log likelihood under a GMRF landscape
    model. 

    Arguments:
    ----------
    f_hat : np.array    
        The graph signal transformed into the Fourier basis. 
    
    eigenvalues : np.ndarray    
        The Graph Laplacian eigenvalues. 
    
    t : float
        The heat diffusion kernel timestep parameter. 
    
    sigma_squared : float
        The empirical variance in the signal. 
    
    epsilon : float, default = `1e-8`.
        Small float for numerical stability.
    
    Returns:
    --------
    log_likelihood : float
        The log likelihood. 
    
    log_det : float
        The log determinant of the Gaussian. 
    
    quadratic form : float
        The qudratic form of the Gaussian. 
    """

    n = len(f_hat)

    # Adjust eigenvalues to avoid zero
    lambda_adjusted = eigenvalues + epsilon

    # Compute heat kernel eigenvalues
    h_i = np.exp(-t * lambda_adjusted)

    # Compute scaling factor
    scaling_factor = (sigma_squared * n) / np.sum(h_i)

    # Scale heat kernel eigenvalues
    h_i_scaled = h_i * scaling_factor

    # Compute inverse of scaled heat kernel eigenvalues
    inv_h_i_scaled = 1 / h_i_scaled

    # Compute quadratic form
    quadratic_form = np.sum(inv_h_i_scaled * (f_hat ** 2))

    # Compute log-determinant
    log_det = np.sum(np.log(h_i_scaled))

    # Compute log-likelihood
    log_likelihood = -0.5 * quadratic_form - 0.5 * log_det - (n / 2) * np.log(2 * np.pi)

    return log_likelihood, log_det, quadratic_form

def generate_sample_H0(sigma_squared: float,
                       t: float,
                       epsilon: float = 1e-8,
                       G: nx.Graph = None,
                       eigenvectors: np.ndarray = None,
                       eigenvalues: np.ndarray = None) -> np.ndarray:
    """
    Function to generate a realization of the GMRF.

    Arguments:
    ----------
    eigenvectors : np.ndarray   
        The Graph Laplacian eigenvectors.

    eigenvalues : np.ndarray    
        The Graph Laplacian eigenvalues. 
    
    t : float
        The heat diffusion kernel timestep parameter. 
    
    sigma_squared : float
        The empirical variance in the signal. 
    
    epsilon : float, default = `1e-8`.
        Small float for numerical stability.

    Returns:
    --------
    sample_H0 : np.ndarray  
        The signal vector sampled under the GMRF.
    """
    if G is not None:
        laplacian, eigenvalues, eigenvectors = compute_laplacian_spectrum(G=G)
    else: 
        assert eigenvalues is not None and eigenvectors is not None
    
    n=len(eigenvalues)
    lambda_adjusted = eigenvalues + epsilon
    h_i = np.exp(-t * lambda_adjusted)
    scaling_factor = (sigma_squared * n) / np.sum(h_i)
    h_i_scaled = h_i * scaling_factor
    Sigma_H0 = eigenvectors @ np.diag(h_i_scaled) @ eigenvectors.T
    sample_H0 = np.random.multivariate_normal(
        mean=np.zeros(eigenvectors.shape[0]),
        cov=Sigma_H0
    )
    return sample_H0

def compute_variances_H0(sigma_squared: float,
                         t: float,
                         epsilon: float = 1e-8,
                         G: nx.Graph = None,
                         eigenvectors: np.ndarray = None,
                         eigenvalues: np.ndarray = None) -> tuple:
    """
    Function to compute the variance vector and covariance matrix of a
    GMRF.

    Arguments:
    ----------
    eigenvectors : np.ndarray   
        The Graph Laplacian eigenvectors.

    eigenvalues : np.ndarray    
        The Graph Laplacian eigenvalues. 
    
    t : float
        The heat diffusion kernel timestep parameter. 
    
    sigma_squared : float
        The empirical variance in the signal. 
    
    epsilon : float, default = `1e-8`.
        Small float for numerical stability.

    Returns:
    --------
    variances_H0 : np.ndarray  
        The variance (the diagonal of the covariance matrix).
    
    Sigma_H0 : np.ndarray
        The covariance matrix. 
    """

    if G is not None:
        laplacian, eigenvalues, eigenvectors = compute_laplacian_spectrum(G=G)
    else: 
        assert eigenvalues is not None and eigenvectors is not None

    n=len(eigenvalues)
    lambda_adjusted = eigenvalues + epsilon
    h_i = np.exp(-t * lambda_adjusted)
    scaling_factor = (sigma_squared * n) / np.sum(h_i)
    h_i_scaled = h_i * scaling_factor
    Sigma_H0 = eigenvectors @ np.diag(h_i_scaled) @ eigenvectors.T
    variances_H0 = np.diag(Sigma_H0)
    return variances_H0, Sigma_H0


