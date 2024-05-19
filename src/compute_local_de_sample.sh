#!/bin/bash
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=128GB
#PBS -l jobfs=100MB
#PBS -l ncpus=64
#PBS -l wd
#PBS -r y
#PBS -l storage=scratch/xz2+g/data/xz2

module purge
module load python3.11.7
deactivate 
source /g/data/xz2/ms2823/venvs/graph_ruggedness_de/bin/activate

python3 -m Compute_Local_Energy_Parallel_Sample -i ../data_files/gb1_dat_comb.csv -p 0.1 -r 10
