import networkx as nx
from graph_fourier_transform import compute_laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans



def transform(fitness: float,
              a: float,
              t: float) -> float: 
    '''
    Funciton to update the fitness value according to the difference
    between a threshold value `t` and the the current fitness value.

    Arguments:
    ----------
    fitness: float
        The current fitness value.
    
    a: float
        The selection pressure on `fitness` that is in effect below
        the threshold value `t`. The scaler of the different in
        `t` - `fitness`.

    t: float
        The threshold value, below which a fitness value `fitness` is
        reduced in the updated fitness value. 
    
    Returns:
    --------
    updated_fitness: float
        The updated fitness value once it has passed through the
        transformation.
    '''

    if fitness > t:
        return fitness
    else:
        g_x = a * ((t - fitness))
        f_x = fitness - g_x
        updated_fitness = max(f_x, 0)

        return updated_fitness
    
def spectral_cluster(G: nx.graph,
                     k: int):
    '''
    Function to perform spectral clustering according to the Fiedler
    method. Performs spectral embedding into the `1:k+1` Laplacian
    eigenvectors and clusters by K-means.

    Arguments:
    ----------
    G: nx.Graph
        The network X landscape graph. Does not require that nodes have
        a `value` attribute.
    
    k: int
        The number of clusters.

    Returns:
    --------
    G_clstr: nx.Graph
        The clustered landscape. 
    '''

    laplacian = compute_laplacian(G=G)
    eigenvalues, eigenvectors = eigsh(laplacian, k=k)
    kmeans = KMeans(n_clusters=k).fit(eigenvectors)
    labels = kmeans.labels_

    G_clstr = G.copy()

    for i, node in enumerate(G.nodes()):
        G_clstr.nodes[node]['spectral_cluster'] = labels[i]
    
    return G_clstr



def count_deltaS(G: nx.Graph,
             t: float):
    '''
    
    '''
    count = 0
    for u, v in G.edges():
        value_u = G.nodes[u]['value']
        value_v = G.nodes[v]['value']
        if (value_u > t and value_v < t) or (value_u < t and value_v > t):
            count += 1
    return count

def count_epsilonS(G: nx.Graph,
                   t: float):
    '''
    '''
    count = 0
    for u, v in G.edges():
        value_u = G.nodes[u]['value']
        value_v = G.nodes[v]['value']
        if value_u > t and value_v > t:
            count += 1
    return count