import random
import numpy as np
from ete3 import Tree
import pyvolve
from tqdm import tqdm
import graph_ruggedness_de


def create_random_tree(num_nodes: int, 
                       mean_length: float,
                       std_dev_length: float) -> Tree:
    '''
    Function to create a random phylogenetic tree using ETE3 package. 

    Arguments:
    ----------
    num_nodes: int
        The number of nodes in the random phylogenetic tree.
    
    mean_length: float
        The average branch-length over the phylogenetic tree.
    
    std_dev_length: float
        The standard deviation of branch-lengths over the phylogenetic
        tree.
    
    Returns:
    --------
    t: ete3.Tree
        Random phylogenetic tree.
    '''

    t = Tree()
    t.populate(num_nodes) 
    for node in t.traverse():
        node.dist = np.random.normal(mean_length, 
                                     std_dev_length)
        
        #Assert all branches have a minimum length of 0.
        while node.dist < 0:
            node.dist = np.random.normal(mean_length, std_dev_length)

    return t


def sequence_evolution(num_nodes: int,
                       mean_branch_length: float,
                       std_dev_branch_length: float,
                       alpha: float,
                       model: str,
                       rate_categories: int,
                       sequence_length: int) -> dict:
    '''
    Function to evolve amino acid sequences over randomised
    phylogenetic trees. Note that parameters (such as alignment) are 
    written to disk in the working directory. 

    Arguments:
    ----------
    num_nodes: int
        The number of nodes in the random phylogenetic tree.
    
    mean_length: float
        The average branch-length over the phylogenetic tree.
    
    std_dev_length: float
        The standard deviation of branch-lengths over the phylogenetic
        tree.
    
    alpha: float
        Alpha shape parameter for rate heterogeneity gamma function. 
    
    moodel: str
        Empirical amino acid replacement matrix. 
    
    rate_categories: int
        The number of rate categories the rate heterogeneity function
        is discretised to. 
    
    sequence_length: int
        The length of the alignment to simulated sequences over. 
    
    Returns:
    --------
    sequences: dict
        The simulated amino acid sequences as the values of a
        dictionary with node names as keys. 
    '''

    t = create_random_tree(num_nodes=num_nodes,
                           mean_length=mean_branch_length,
                           std_dev_length=std_dev_branch_length)
    t = pyvolve.read_tree(tree = t.write(format=5))
    model = pyvolve.Model(model, 
                          alpha=alpha, 
                          num_categories=rate_categories) 
    
    partition = pyvolve.Partition(models=model, 
                                  size=sequence_length)

    evolver = pyvolve.Evolver(partitions=partition, 
                              tree=t)
    evolver()
    sequences = evolver.get_sequences()
    
    return sequences

def sample_sequences(sample_size: int) -> tuple:
    '''
    Function to simulate sequence evolution over phylogenetic trees 
    with randomly sampled evolutionary parameters.

    Arguments:
    ----------
    sample_size: int
        The number of independent replicates to perform. 
    
    Returns:
    --------
    dir_dict: dict
        Dictionary of dictionaries with replicate index as key and
        dirichlet energy (global) as values. 
    
    sample_dict: dict
        Dictionary of dictionaries with replicate index as key and
        sequence evolution parameters as values. 
    '''

    dir_dict = {}
    sample_dict = {}

    for _ in tqdm(enumerate(range(sample_size))):
        try:

            num_nodes = int(np.random.uniform(50, 200))
            mean_branch_length = np.random.uniform(0.01, 0.5)
            std_dev_branch_length = np.random.uniform(0.001, 0.1)
            alpha = np.random.uniform(0.1, 0.5)
            model = random.choice(('WAG', 'LG'))
            rate_categoies = 4
            sequence_length = int(np.random.uniform(100, 600))

            sample_info = {
                'num_nodes' : num_nodes,
                'mean_branch_length' : mean_branch_length,
                'std_dev_branch_length' : std_dev_branch_length,
                'alpha' : alpha,
                'model' : model,
                'sequence_length' : sequence_length
            }

            seq_dict = sequence_evolution(
                num_nodes=num_nodes,
                mean_branch_length=mean_branch_length,
                std_dev_branch_length=std_dev_branch_length,
                alpha=alpha,
                model=model,
                rate_categories=rate_categoies,
                sequence_length=sequence_length
                )
            seq_ls = list(seq_dict.values())
            values = [0]*len(seq_ls)
            G = graph_ruggedness_de.build_ohe_graph(seq_ls=seq_ls,
                                                values=values)
            eign_dir_dict = {}
            for eign in list(range(50)):
                G = graph_ruggedness_de.compute_elementary_landscape(G=G, n=eign)
                dir_energy = graph_ruggedness_de.compute_dirichlet_energy(G=G)
                eign_dir_dict[eign] = dir_energy
            
            dir_dict[_] = eign_dir_dict
            sample_dict[_] = sample_info
        
        except:
            continue
    
    return dir_dict, sample_dict