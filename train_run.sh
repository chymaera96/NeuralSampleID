#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 16          # 16 cores per GPU
#$ -l h_rt=30:0:0    # 40 hours runtime
#$ -l h_vmem=11G      # 11G RAM per core
#$ -l gpu=2           # request 2 GPU
#$ -l gpu_type=ampere
#$ -l rocky
#$ -N nsid_tc_22
#$ -o /data/home/acw723/NeuralSampleID/hpc_out
#$ -m beas

module load python/3.12.1-gcc-12.2.0 
source ../grafp2_venv/bin/activate
sh resnet_script.sh train
