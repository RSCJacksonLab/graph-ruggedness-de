import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh

def compute_laplacian_spectrum(G: nx.Graph):
    '''
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
    '''
    
    laplacian = nx.normalized_laplacian_matrix(G, weight='inv_weight').asfptype()
    L_dense = laplacian.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
    
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return laplacian, eigenvalues, eigenvectors


def standardise_likelihoods(log_likelihood: float,
                            sigma_squared: float,
                            t: float,
                            epsilon: float = 1e-8,
                            G: nx.Graph = None,
                            eigenvectors: np.ndarray = None,
                            eigenvalues: np.ndarray = None,
                            samples: int = 100):
    '''
    
    '''

    #Perform spectral decomposition of Laplacian only once.
    if G is not None:
        laplacian, eigenvalues, eigenvectors = compute_laplacian_spectrum(G=G)
    else: 
        assert eigenvalues is not None and eigenvectors is not None


    # List to store likelihoods
    h0_likelihoods = []

    for _ in range(samples):
        # Generate a sample under H0
        h0_sample = generate_sample_H0(sigma_squared = sigma_squared,
                                       t=t,
                                       epsilon=epsilon,
                                       eigenvalues=eigenvalues,
                                       eigenvectors=eigenvectors)
        #GFT of sample
        f_hat = eigenvectors.T @ h0_sample

        sigma_squared_sample = np.var(h0_sample, ddof=1)

        # Compute the log-likelihood of sample under H0
        h0_likelihood,  _, _ = compute_log_likelihood_H0(f_hat=f_hat,
                                                         eigenvalues=eigenvalues,
                                                         t=t,
                                                         sigma_squared=sigma_squared_sample,
                                                         epsilon=epsilon)
        
        h0_likelihoods.append(h0_likelihood)

    #Compute Z-score of empirical log likelihood.
    z_score = np.abs((log_likelihood - np.mean(h0_likelihoods)) / np.std(h0_likelihoods, ddof=1))

    return z_score


def compute_log_likelihood_H0(f_hat,
                              eigenvalues,
                              t,
                              sigma_squared,
                              epsilon=1e-8,
                              ):


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
                       eigenvalues: np.ndarray = None):
    '''
    
    '''
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
                         eigenvalues: np.ndarray = None):
    '''
    
    '''

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


def compute_gmrf_ruggedness(G: nx.Graph,
                            t0: float = None,
                            standardise: bool = False,
                            samples: int = None):
    '''

    '''
    laplacian, eigenvalues, eigenvectors = compute_laplacian_spectrum(G=G)
    
    signal = np.array([G.nodes[node]['value'] for node in G.nodes()])
    sigma_squared = np.var(signal, ddof=1)
    #Normalise signal so that the average = 0
    mu = np.average(signal)
    signal = signal - mu

    #Compute GFT
    f_hat = eigenvectors.T @ signal

    log_likelihood_H0, log_det, quadratic_form = compute_log_likelihood_H0(f_hat=f_hat,
                                                                           eigenvalues=eigenvalues,
                                                                           t=t0,
                                                                           sigma_squared=sigma_squared)
    
    if standardise:
        
        assert samples is not None

        signal = np.array([G.nodes[node]['value'] for node in G.nodes()])
        sigma_squared = np.var(signal, ddof=1)

        h0_likelihoods = []

        for _ in range(samples):

            # Generate a sample under H0
            h0_sample = generate_sample_H0(
                G=G,
                t=t0,
                sigma_squared=sigma_squared
            )

            G_cpy = G.copy()
            for i, node in enumerate(G_cpy.nodes()):
                G_cpy.nodes[node]['value'] = h0_sample[i]
            
            # Recursively compute the log-likelihood under H0
            h0_likelihood, _, _ = compute_gmrf_ruggedness(G=G_cpy,
                                                          t0=t0,
                                                          standardise=False)
            h0_likelihoods.append(h0_likelihood)

        z_score = np.abs((log_likelihood_H0 - np.mean(h0_likelihoods)) / np.std(h0_likelihoods, ddof=1))
        
        return z_score
    
    else:
        return log_likelihood_H0, log_det, quadratic_form


##Wrappers for H1 

def compute_log_likelihood_H1(f_hat,
                              eigenvalues,
                              t1,
                              sigma_squared,
                              epsilon=1e-8,
                              standardised: bool = True,
                              draw_samples: bool = True,
                              eigenvectors = None,
                              replicates: int = 10
                              ):
    '''
    
    '''
    return compute_log_likelihood_H0(f_hat=f_hat,
                              eigenvalues=eigenvalues,
                              t=t1,
                              sigma_squared=sigma_squared,
                              epsilon=epsilon,
                              standardised=standardised,
                              draw_samples=draw_samples,
                              eigenvectors=eigenvectors,
                              replicates=replicates
                              )


def generate_sample_H1(G:nx.Graph,
                       t1: float,
                       sigma_squared: float=None):
    '''
    
    '''
    return generate_sample_H0(G=G,
                              sigma_squared=sigma_squared,
                              t=t1)

def compute_marginal_variances_H1(G:nx.Graph,
                                  t1: float,
                                  sigma_squared: float=None):
    '''
    
    '''
    return compute_marginal_variances_H0(G=G,
                                         sigma_squared=sigma_squared,
                                         t=t1)