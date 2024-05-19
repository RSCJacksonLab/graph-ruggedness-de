import pandas as pd
import argparse
import graph_ruggedness_de
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

def main():
    os.makedirs('results', exist_ok=True)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-i',
        '--input_csv',
        type=str,
        help='Input CSV file to process.'
    )

    parser.add_argument(
        '-p',
        '--sampling_prop',
        type=float,
        help='Graph sub-ampling proportion.'
    )

    parser.add_argument(
        '-r',
        '--replicates',
        type=int,
        help='Number of subsampling replicates.'
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    seq_ls = df['sequences'].tolist()
    values = df['fitness'].tolist()
    scaler = MinMaxScaler()
    values = [val[0] for val in (scaler.fit_transform(np.array(values).reshape(-1,1)))]

    G_k = graph_ruggedness_de.build_ohe_graph(seq_ls=seq_ls,
                                            values=values,
                                            edges=False,
                                            hamming_edges=False, 
                                            approximate=True,
                                            n=400)

    sampling_prop = args.sampling_prop
    for i in args.replicates:

        G_sampled, sampled_nodes, sampled_values = graph_ruggedness_de.sample_graph(G=G_k,
                                                                                    sample_size=sampling_prop)
        graph_ruggedness_de.add_ohe_knn_edges_approx(G=G_sampled,
                                                        k=int(np.sqrt(G_sampled.number_of_nodes())))
        graph_ruggedness_de.compute_local_dirichlet_energy_parallel(G=G_sampled,
                                                                    approximate=True)
        local_de_dict = {
            node : node['local_dirichlet'] for node in G_sampled.nodes()
        }
        local_de_df = pd.DataFrame({
            'node': local_de_dict.keys(),
            'local_de_energy': local_de_dict.values()
        })
        file_name = os.path.splitext(os.path.basename(args.input_csv))[0]
        local_de_df.to_csv(f'results/{file_name}_rep{i}_prop{sampling_prop}.csv')

if __name__ == '__main__':
    main()