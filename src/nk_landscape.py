import networkx as nx
import random
import numpy as np

def generate_nk_landscape_amino_acids(N: int,
                                      K: int,
                                      amino_acids: list=None,
                                      max_size: int=np.inf) -> nx.Graph:
    '''
    Generates an NK landscape for amino acid sequences and returns it
    as a NetworkX graph.

    Arguments:
    ----------
    N: int
        The number of amino aicd poisition in the sequence
    
    k: int
        The number of interactions per position in the sequence.
    
    amino_acids: list[str]
        The list of amino acids to use. Default is the 20 canonical
        amino acids.
    
    max_size: int
        The maximum graph size. Default is inf.

    Returns:
    --------
    G: networkX.Graph
        The generated NK landscape hamming graph.
    '''
    
    if amino_acids is None:
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
                       'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                       'T', 'W', 'Y', 'V']
    A = len(amino_acids)  # Alphabet size

    if K >= N:
        raise ValueError("K must be less than N.")
    aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

    interactions = []
    for i in range(N):
        possible_partners = list(range(N))
        possible_partners.remove(i)
        partners = random.sample(possible_partners, K)
        interactions.append(partners)

    contributions = []
    for i in range(N):
        num_states = A ** (K + 1)
        fitness_values = [random.random() for _ in range(num_states)]
        contributions.append(fitness_values)

    def calculate_fitness(sequence):
        total_fitness = 0.0
        for i in range(N):
            positions = [i] + interactions[i]
            state_indices = [aa_to_index[sequence[pos]] for pos in positions]

            state = 0
            for idx in state_indices:
                state = state * A + idx
            fitness = contributions[i][state]
            total_fitness += fitness
        average_fitness = total_fitness / N
        return average_fitness

    num_samples = min(max_size, A ** N)  # Adjust number of samples
    sequences = set()
    while len(sequences) < num_samples:
        seq = ''.join(random.choices(amino_acids, k=N))
        sequences.add(seq)
    sequences = list(sequences)

    G = nx.Graph()

    for sequence in sequences:
        average_fitness = calculate_fitness(sequence)
        # Add the node with its fitness value
        G.add_node(sequence, value=average_fitness)

    # Add edges between sequences that differ by one mutation
    for seq1 in sequences:
        for i in range(N):
            for aa in amino_acids:
                if aa != seq1[i]:
                    seq_list = list(seq1)
                    seq_list[i] = aa
                    seq2 = ''.join(seq_list)
                    if seq2 in G:
                        G.add_edge(seq1, seq2)

    return G