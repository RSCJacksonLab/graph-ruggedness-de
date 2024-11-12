import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh

def compute_laplacian_spectrum(G: nx.Graph,
                               k: int = None):
    '''
    Computes the eigenvalues and eigenvectors of the Laplacian matrix
    of graph G.
    
    Arguments:
    -----------
    G : networkx.Graph
        The input graph.
    k : int or None
        Number of eigenvalues and eigenvectors to compute. If `None`,
        compute all.
    
    Returns:
    --------
    laplacian : scipy.sparse matrix
        The Laplacian matrix of the graph.

    eigenvalues : np.ndarray
        The computed eigenvalues.

    eigenvectors : np.ndarray
        The computed eigenvectors.
    '''
    
    # Compute the Laplacian matrix in sparse format
    laplacian = nx.normalized_laplacian_matrix(G).asfptype()
    
    n = G.number_of_nodes()
    if k is not None and k < n:
        #Compute first k eigenvectors/values of the laplacian.
        eigenvalues, eigenvectors = eigsh(laplacian, k=k, which='SM')
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
    
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return laplacian, eigenvalues, eigenvectors


def compute_log_bayes_factor(log_likelihood_H0: float,
                             log_likelihood_H1: float):
    '''
    Computes the logarithm of the Bayes factor between H1 and H0.
    
    Arguments:
    -----------
    log_likelihood_H0: float
        The log-likelihood under H0.
    
    log_likelihood_H1: float
        The log-likelihood under H1.
    
    Returns:
    --------
    log_BF: float
        The logarithm of the Bayes factor.
    '''
    log_BF = log_likelihood_H1 - log_likelihood_H0
    return log_BF

def compute_log_likelihood_H0(f_hat, eigenvalues, t, epsilon=1e-8):
    # Avoid zero eigenvalue by adding epsilon
    lambda_adjusted = eigenvalues + epsilon
    
    # Compute unnormalized heat kernel eigenvalues
    h_i = np.exp(-t * lambda_adjusted)
    
    # Compute the sum of the heat kernel eigenvalues
    sum_h_i = np.sum(h_i)
    
    # Normalize the heat kernel eigenvalues
    h_i_normalized = h_i / sum_h_i
    
    # Compute the inverse of the normalized heat kernel eigenvalues
    inv_h_i_normalized = 1 / h_i_normalized  # Equivalent to inv_h_i_normalized = sum_h_i / h_i
    
    # Compute the quadratic form
    quadratic_form = np.sum(inv_h_i_normalized * (f_hat ** 2))
    
    # Compute the log-determinant of the normalized heat kernel
    # Original log-determinant
    log_det = np.sum(np.log(h_i))
    
    # Adjusted log-determinant after normalization
    n = len(f_hat)
    log_det_normalized = log_det - n * np.log(sum_h_i)
    
    # Compute log-likelihood
    log_likelihood = -0.5 * quadratic_form + 0.5 * log_det_normalized - (n / 2) * np.log(2 * np.pi)
    return log_likelihood

def compute_log_likelihood_H1(f):

    n = len(f)
    sigma_squared = np.var(f, ddof=1)
    quadratic_form = np.sum(f ** 2) / sigma_squared
    log_det = n * np.log(sigma_squared)
    log_likelihood = -0.5 * quadratic_form - 0.5 * log_det - (n / 2) * np.log(2 * np.pi)
    return log_likelihood

def compute_gmrf_ruggedness(G: nx.Graph,
                            t: int):
    '''
    Function to determine the ruggedness of the graph `G` as the log
    Bayes factor under H0 and H1. The priors for H0 and H1 are
    defined by a Gaussian Markov random fields. Under H0, the 
    covariation matrix is defined by the `k` lowest frequency GFT
    coefficients and Laplacian eigenvalues (such that covariation is
    expected only where the signal is smooth over the graph). The H1
    prior is defined by the variance of the data and is therefore not
    constrained to a graph structure. The Bayes factor is a measure of
    whether the evidence supports the GFT coefficients of the empirical
    graph belonging to a smooth distirubtion defined under H0 or a
    rugged distribution defined under H1.

    Arguments:
    ----------
    g: nx.Graph
        The graph to be analyzed. 
    
    k: int, default=`1`
        The index of the low-frequency GFT coefficients and Laplacian
        eigenvalues to use. 
    
    Returns:
    --------
    log_bf: float
        The log of the Bayes factor.
    '''



    laplacian, eigenvalues, eigenvectors = compute_laplacian_spectrum(G=G,
                                                                      k=None)
    
    signal = np.array([G.nodes[node]['value'] for node in G.nodes()])
    #Normalise signal so that the average = 0
    mu = np.average(signal)
    signal = signal - mu

    #Compute GFT
    f_hat = eigenvectors.T @ signal
    log_likelihood_H0 = compute_log_likelihood_H0(f_hat=f_hat,
                                                  eigenvalues=eigenvalues,
                                                  t=t)
    
    log_likelihood_H1 = compute_log_likelihood_H1(f=signal)

    log_bf = compute_log_bayes_factor(log_likelihood_H0=log_likelihood_H0,
                                      log_likelihood_H1=log_likelihood_H1)
    
    
    return log_bf, log_likelihood_H0, log_likelihood_H1
    


######OBSOLETE########

# def compute_log_likelihood_H0(f_hat: np.ndarray,
#                               eigenvalues: np.ndarray,
#                               k: int=1):

#     '''
#     Computes the log-likelihood under H0. Does not explicitly require
#     sigma0 as quadratic form of sigma0 in the spectral domain
#     simplifies to the sum of the eigenvalues dot f_hat **2 over the `k`
#     low frequency indices.
    
#     Arguments:
#     -----------
#     f_hat: np.ndarray
#         The graph Fourier transform coefficient vector.

#     eigenvalues: np.ndarray
#         The eigenvalues of the graph laplacian.

#     k : int, default=`1`
#         The cutoff index for low-frequency components.
    
#     Returns:
#     --------
#     log_likelihood_H0: float
#         The log-likelihood of sampling `f_hat` under H0.
#     '''

#     m = k - 1
    
#     #Include only the `k` lowest frequency modes, without the Fiedler
#     #vector / value.
#     lambda_low = eigenvalues[1:k]
#     f_hat_low = f_hat[1:k]
    
#     #Gaussian multivariate standard equation
#     exponent = -0.5 * np.sum(lambda_low * np.abs(f_hat_low) ** 2)
    
#     #Compute the log determinant of the covariance matrix
#     log_det_Sigma0 = -np.sum(np.log(lambda_low))
    
#     #Gaussian multivariate standard equation
#     normalization = -0.5 * log_det_Sigma0 - (m / 2) * np.log(2 * np.pi)
    
#     log_likelihood_H0 = exponent + normalization
    
#     return log_likelihood_H0




# def compute_log_likelihood_H1(f_hat: np.ndarray): ####Variation based
#     '''
#     Computes the log-likelihood under H1. Under H1, the GMRF covariance
#     matrix is simply variance dot Identity matrix.
    
#     Arguments:
#     -----------
#     f_hat: np.ndarray
#         The GFT coefficient vector.
    
#     Returns:
#     --------
#     log_likelihood_H1: float
#         The log-likelihood of sampling `f_hat` under H1.
#     '''
#     n = len(f_hat)
#     #Covariance matrix for H1 is the variance matrix.
#     sigma_squared = np.var(f_hat, ddof=1)
    
#     #Standard multivariate Gaussian
#     exponent = -0.5 * np.sum(np.abs(f_hat) ** 2) / sigma_squared
    
#     #Eigenvalues of of sigma @ I are sigma.
#     log_det_Sigma1 = n * np.log(sigma_squared)

#     #Standard multivariate Gaussian
#     normalization = -0.5 * log_det_Sigma1 - (n / 2) * np.log(2 * np.pi)
    
#     log_likelihood_H1 = exponent + normalization
#     return log_likelihood_H1




# def compute_log_likelihood_H1(f_hat: np.ndarray,
#                               eigenvalues: np.ndarray):
    
#     '''
    
#     '''
#     n = len(f_hat)
#     m = n - 1  # Exclude zero eigenvalue

#     lambda_nonzero = eigenvalues[1:]
#     f_hat_nonzero = f_hat[1:]

#     exponent = -0.5 * np.sum(lambda_nonzero * np.abs(f_hat_nonzero) ** 2)
#     log_det_Sigma1 = -np.sum(np.log(lambda_nonzero))
#     normalization = -0.5 * log_det_Sigma1 - (m / 2) * np.log(2 * np.pi)
#     log_likelihood_H1 = exponent + normalization

#     return log_likelihood_H1

