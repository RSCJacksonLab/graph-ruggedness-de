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

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    seq_ls = df['sequences'].tolist()
    values = df['fitness'].tolist()
    scaler = MinMaxScaler()
    values = [val[0] for val in (scaler.fit_transform(np.array(values).reshape(-1,1)))]

    G_k = graph_ruggedness_de.build_ohe_graph(seq_ls=seq_ls,
                                            values=values,
                                            edges=True,
                                            hamming_edges=False, 
                                            approximate=True,
                                            n=400)
    graph_ruggedness_de.compute_local_dirichlet_energy_parallel(G=G_k,
                                                                approximate=True)
    local_de_dict = {
        node : node['local_dirichlet'] for node in G_k.nodes()
    }
    local_de_df = pd.DataFrame({
        'node': local_de_dict.keys(),
        'local_de_energy': local_de_dict.values()
    })
    file_name = os.path.splitext(os.path.basename(args.input_csv))[0]
    local_de_df.to_csv(f'results/{file_name}_de.csv')

if __name__ == '__main__':
    main()
