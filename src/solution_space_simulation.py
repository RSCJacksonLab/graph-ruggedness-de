import networkx as nx
from gaussian_markov_random_field import compute_laplacian_spectrum
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import random


def spectral_cluster(G: nx.Graph,
                     k: int) -> nx.Graph:
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
    laplacian = nx.laplacian_matrix(G, weight='inv_weight').asfptype()
    eigenvalues, eigenvectors = eigsh(laplacian, k=k+1, which='SM')
    embedding = eigenvectors[:, 1:]  # Exclude the first eigenvector
    kmeans = KMeans(n_clusters=k).fit(embedding)
    labels = kmeans.labels_

    G_clstr = G.copy()
    for i, node in enumerate(G.nodes()):
        G_clstr.nodes[node]['spectral_cluster'] = labels[i]
    
    return G_clstr

def assign_clustered_values(G: nx.Graph) -> nx.Graph:
    '''
    Assigns 'value' attribute to nodes based on their spectral cluster.
    Selects one cluster randomly and assigns value 1 to its nodes,
    and 0 to others.

    Arguments:
    ----------
    G: nx.Graph
        The graph with 'spectral_cluster' attribute assigned to nodes.

    Returns:
    --------
    G: nx.Graph
        The graph with 'value' attribute assigned to nodes.
    '''
    clstrs = set([node[1]['spectral_cluster'] for node in G.nodes(data=True)])
    clstr = random.choice(list(clstrs))

    for node in G.nodes(data=True):
        if node[1]['spectral_cluster'] == clstr:
            G.nodes[node[0]]['value'] = 1
        else:
            G.nodes[node[0]]['value'] = 0

    return G


def count_deltaS(G: nx.Graph) -> int:
    '''
    Function to cound the number of edges in deltaS.
    
    Arguments:
    ----------
    G : nx.Graph
        The fitness landscape graph.
    
    Returns:
    --------
    count : int
        The number of edges in delta S.
    '''
    count = 0
    for u, v in G.edges():
        value_u = G.nodes[u]['value']
        value_v = G.nodes[v]['value']
        if (value_u == 1 and value_v == 0) or (value_u == 0 and value_v == 1):
            count += 1
    return count

def count_epsilonS(G: nx.Graph) -> int:
    '''
    Function to cound the number of edges in epsilon S.
    
    Arguments:
    ----------
    G : nx.Graph
        The fitness landscape graph.
    
    Returns:
    --------
    count : int
        The number of edges in epsilon S.
    '''
    count = 0
    for u, v in G.edges():
        value_u = G.nodes[u]['value']
        value_v = G.nodes[v]['value']
        if value_u == 1 and value_v == 1:
            count += 1
    return count