import numpy as np
import networkx as nx
from scipy.optimize import minimize_scalar
from gaussian_markov_random_field import (
    compute_log_likelihood_H0,
    precompute_GMRF_stats)


def fit_t_bayesian_laplace(G: nx.Graph,
                           t_min: float = 0.01,
                           t_max: float = 1000.0,
                           epsilon: float = 1e-8,
                           verbose: bool = False) -> tuple:
    """
    Function to estimate the Posterior probability distribution of t
    using the Laplace approximation.

    Arguments:
    ----------
    G : nx.Graph
        The fitness landscape graph. 

    t_min : float
        The prior lower bound on t. 
    
    t_max : float
        The prior upper bound on t. 
    
    epsilon : float
        Small float for numerical stability.
    
    verbose : bool, default=`False`
        Boolean for verbose output. 

    Returns:
    --------
    t_map : float
        The maximum a posteriori t value.
    
    ci_lower : float
        The lower bound of the confidence interval on t.
    
    ci_upper : float
        The upper bound of the confidence interval on t.
    
    logpost_map: float
        The log posterior probability of the MAP t value.
    
    var_approx : float
        The variance approximated as the second derivative of 
        the negative log posterior with respect to t. 
    """

    laplacian = nx.normalized_laplacian_matrix(G, weight='inv_weight').asfptype()
    L_dense = laplacian.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L_dense)

    # Sort by ascending eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    fiedler_value = eigenvalues[1]
    eigenvectors = eigenvectors[:, idx]

    signal = np.array([G.nodes[node]['value'] for node in G.nodes()])
    mu = np.mean(signal)
    signal_centered = signal - mu

    f_hat = eigenvectors.T @ signal_centered

    sigma_squared = np.var(signal_centered, ddof=1)

    def neg_log_posterior(t: float) -> float:
        """
        Helper function to return negative log posterior.

        Arguments:
        ----------
        t : float
            The timestep of the heat kernel. 
        
        Returns:
        --------
        float
            The negative log posterior probability.
        """
        if t < t_min or t > t_max:
            return np.inf

        log_pri = 0.0
        
        ll = compute_log_likelihood_H0(
            f_hat=f_hat,
            eigenvalues=eigenvalues,
            t=t,
            sigma_squared=sigma_squared,
            epsilon=epsilon
        )[0]

        return -(ll + log_pri)
    
    result = minimize_scalar(
        neg_log_posterior,
        bounds=(t_min, t_max),
        method='bounded',
        options={'maxiter': 200}
    )

    t_map = result.x
    if verbose:
        print(f"Optimization success: {result.success}, t_map={t_map:.4f}, f(t_map)={result.fun:.4f}")

    h = 1e-4 * max(1.0, abs(t_map))  # step size ~ 1e-4 * scale_of_t
    t_minus = max(t_min, t_map - h)
    t_plus  = min(t_max, t_map + h)

    f0 = neg_log_posterior(t_map)
    f_minus = neg_log_posterior(t_minus)
    f_plus  = neg_log_posterior(t_plus)

    second_deriv = (f_plus - 2.0*f0 + f_minus) / ( (t_plus - t_minus)/2.0 )**2

    if second_deriv <= 0:
        if verbose:
            print("Warning: second derivative <= 0, cannot do standard Laplace. Setting it to a small positive.")
        second_deriv = 1e-12

    var_approx = 1.0 / second_deriv
    std_approx = np.sqrt(var_approx)

    # 95% credible interval (approx Gaussian)
    ci_lower = t_map - 1.96 * std_approx
    ci_upper = t_map + 1.96 * std_approx

    # Clip to [t_min, t_max]
    ci_lower = max(ci_lower, t_min)
    ci_upper = min(ci_upper, t_max)

    logpost_map = -(f0) 

    return t_map, ci_lower, ci_upper, logpost_map, var_approx



def compute_log_evidence_single(G: nx.Graph,
                                t_min=0.01,
                                t_max=100.0,
                                epsilon=1e-8,
                                verbose=False):
    """
    Function to compute the marginal likelihood (log evidence) of a 
    single Graph using the Laplace approximation. 

    Arguments:
    ----------
    G : nx.Graph
        The fitness landscape graph. 

    t_min : float
        The prior lower bound on t. 
    
    t_max : float
        The prior upper bound on t. 
    
    epsilon : float
        Small float for numerical stability.
    
    verbose : bool, default=`False`
        Boolean for verbose output. 

    Returns:
    --------
    log_evidence : float
        The marginal log likelihood.
    """
    t_map, ci_lower, ci_upper, logpost_map, var_approx = fit_t_bayesian_laplace(
        G, t_min=t_min, t_max=t_max, epsilon=epsilon, verbose=verbose)
    log_evidence = logpost_map + 0.5 * np.log(2 * np.pi * var_approx)
    return log_evidence

def compute_joint_log_evidence(G1: nx.Graph,
                               G2: nx.Graph,
                               t_min=0.01,
                               t_max=100.0,
                               epsilon=1e-8,
                               verbose=False):
    """
    Function to compute the marginal likelihood of a joint fitness
    diffusion process over two fitness landscapes.
    
    Arguments:
    ----------
    G1 : nx.Graph
        The first pairwise fitness landscape graph. 

    G2 : nx.Graph
        The second pairwise fitness landscape graph. 

    t_min : float
        The prior lower bound on t. 
    
    t_max : float
        The prior upper bound on t. 
    
    epsilon : float
        Small float for numerical stability.
    
    verbose : bool, default=`False`
        Boolean for verbose output. 

    Returns:
    --------
    log_evidence_joint : float
        The joint marginal likelihood. 
    """

    # Precompute stats for each graph.
    f_hat1, eigenvalues1, sigma_squared1 = precompute_GMRF_stats(G1, epsilon)
    f_hat2, eigenvalues2, sigma_squared2 = precompute_GMRF_stats(G2, epsilon)
    
    def neg_log_post_joint(t: float) -> float:
        """
        Helper function
        """
        if t < t_min or t > t_max:
            return np.inf
        # Compute log likelihoods for each graph.
        ll1 = compute_log_likelihood_H0(
            f_hat=f_hat1,
            eigenvalues=eigenvalues1,
            t=t,
            sigma_squared=sigma_squared1,
            epsilon=epsilon
        )[0]
        ll2 = compute_log_likelihood_H0(
            f_hat=f_hat2,
            eigenvalues=eigenvalues2,
            t=t,
            sigma_squared=sigma_squared2,
            epsilon=epsilon
        )[0]

        return -(ll1 + ll2)
    
    # Optimize the joint negative log posterior.
    result = minimize_scalar(
        neg_log_post_joint,
        bounds=(t_min, t_max),
        method='bounded',
        options={'maxiter': 300}
    )
    t_joint_map = result.x
    if verbose:
        print(f"Joint optimization success: {result.success}, t_joint_map={t_joint_map:.4f}, f(t_joint_map)={result.fun:.4f}")
    
    # Estimate the second derivative at t_joint_map via finite differences.
    h = 1e-4 * max(1.0, abs(t_joint_map))
    t_minus = max(t_min, t_joint_map - h)
    t_plus  = min(t_max, t_joint_map + h)
    f0 = neg_log_post_joint(t_joint_map)
    f_minus = neg_log_post_joint(t_minus)
    f_plus  = neg_log_post_joint(t_plus)
    second_deriv = (f_plus - 2.0 * f0 + f_minus) / (((t_plus - t_minus) / 2.0) ** 2)
    if second_deriv <= 0:
        if verbose:
            print("Warning: joint second derivative <= 0. Setting to small positive value.")
        second_deriv = 1e-12
    var_joint = 1.0 / second_deriv
    
    log_evidence_joint = -f0 + 0.5 * np.log(2 * np.pi * var_joint)
    return log_evidence_joint

def compute_bayes_factor(G1: nx.Graph,
                         G2: nx.Graph,
                         t_min=0.01,
                         t_max=1000.0,
                         epsilon=1e-8,
                         verbose=False):
    """
    Function to compute the Bayes Factor between two fitness landscapes
    sharing a joint vs. independent fitness diffusion processes.

    Arguments:
    ----------
    G1 : nx.Graph
        The first pairwise fitness landscape graph. 

    G2 : nx.Graph
        The second pairwise fitness landscape graph. 

    t_min : float
        The prior lower bound on t. 
    
    t_max : float
        The prior upper bound on t. 
    
    epsilon : float
        Small float for numerical stability.
    
    verbose : bool, default=`False`
        Boolean for verbose output. 

    Returns:
    --------
    BF : float
        The Bayes factor.
    
    log_evidence_H0 : float
        The log evidence under H0. 
    
    log_evidence_H1 : float
        The log evidence under H1.
    """
    log_evidence1 = compute_log_evidence_single(G1, t_min, t_max, epsilon, verbose)
    log_evidence2 = compute_log_evidence_single(G2, t_min, t_max, epsilon, verbose)
    log_evidence_H1 = log_evidence1 + log_evidence2

    log_evidence_H0 = compute_joint_log_evidence(G1, G2, t_min, t_max, epsilon, verbose)
    
    BF = np.exp(log_evidence_H0 - log_evidence_H1)
    
    if verbose:
        print("Log Evidence (H0):", log_evidence_H0)
        print("Log Evidence (H1):", log_evidence_H1)
        print("Bayes Factor (BF):", BF)
    
    return BF, log_evidence_H0, log_evidence_H1
