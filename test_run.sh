#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8          # cores per GPU
#$ -l h_rt=36:0:0    # hours runtime
#$ -l h_vmem=11G      #  RAM per core
#$ -l gpu=1           # request GPU
#$ -l gpu_type=ampere
#$ -N grafp_tc29_medeval
#$ -o /data/home/acw723/GraFP/hpc_out
#$ -m beas

module load python/3.10.7
source ../grafp_venv/bin/activate
python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=10 --text=1000_fma_medium_10