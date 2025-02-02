if __name__ == "__main__":

    import os 
    import pandas as pd
    import numpy as np 
    import matplotlib.pyplot as plt
    import graph_ruggedness_de
    import gaussian_markov_random_field
    import networkx as nx
    from pathlib import Path

    file_list = os.listdir('../data_files/megascale_folding/')
    file_list = [file for file in file_list if file != '.DS_Store']

    file_list.sort()  # Ensure consistent order

    likelihood_values = []
    files = []
    fielder_values = []
    norm_factor = []

    replicates = 1

    landscape_likelihoods = {}

    if 'folding_data_checkpoint_hamming.csv' in os.listdir('./'):
        checkpoint_df = pd.read_csv('folding_data_checkpoint.csv', index_col=0)
        parsed_files = checkpoint_df['dataset'].tolist()

    else:
        checkpoint_df = pd.DataFrame({
            'dataset': [],
            'z_score': [],
            'log_likelihood': [],
            'fiedler_value': []
        })
        parsed_files = []

    for idx, file in enumerate(file_list):        
        try:
            if file.startswith('_'):
                continue
            elif file in parsed_files:
                continue
            # ax_col is an array of two axes: [hist_ax, bar_ax]

            df = pd.read_csv(f'../data_files/megascale_folding/{file}')
            df = df.dropna()
            
            # Continue if dataset too large
            if len(df) > 4000:
                continue

            # Add wt sequences and 0 value.
            wt_mut = df.iloc[0,0]
            wt_seq = df.iloc[0,1]
            wt_list =list(wt_seq)
            wt_list[int(wt_mut[1:-1])-1] = wt_mut[0]
            wt_seq = ''.join(wt_list)
            
            seq_ls = df['mutated_sequence'].tolist()
            seq_ls.append(wt_seq)
            values = df['DMS_score'].tolist()
            values.append(0)

            # Build graph and compute metrics
            G_k = graph_ruggedness_de.build_ohe_graph(
                seq_ls=seq_ls,
                values=values,
                edges=False,
            )

            #Add single mutation edges
            graph_ruggedness_de.add_hamming_edges(G_k, threshold=1)

            comb_laplacian = nx.laplacian_matrix(G_k).asfptype()
            comb_L_dense = comb_laplacian.toarray()
            comb_eigenvalues, _ = np.linalg.eigh(comb_L_dense)
            comb_eigenvalues.sort()
            fielder_value = comb_eigenvalues[1]

            if fielder_value < 1:
                continue
            
            fielder_values.append(fielder_value)


            z_score,log_likelihood = gaussian_markov_random_field.compute_gmrf_ruggedness(G=G_k,
                                                                                t0=30 / fielder_value,
                                                                                standardise=True,
                                                                                samples=10)
            checkpoint_df.loc[idx] = [file, z_score, log_likelihood, fielder_value]

            # Write a checkpoint csv.
            if idx % 2 == 0:
                checkpoint_df.to_csv('folding_data_checkpoint_hamming.csv')
            
            parsed_files.append(file)
            landscape_likelihoods[file] = (z_score, log_likelihood)
        except Exception as e:
            print(f'Error occurred: {str(e)}')




