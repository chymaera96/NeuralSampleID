#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 16          # 16 cores per GPU
#$ -l h_rt=20:0:0    # 40 hours runtime
#$ -l h_vmem=11G      # 11G RAM per core
#$ -l gpu=2           # request 2 GPU
#$ -l gpu_type=ampere
#$ -N nsid_tc_16
#$ -o /data/home/acw723/NeuralSampleID/hpc_out
#$ -m beas

module load python/3.10.7
source ../grafp_venv/bin/activate
module load gcc/12.1.0
python train.py --ckp=tc_16



