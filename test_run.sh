!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 24          # 16 cores per GPU
#$ -l h_rt=1:0:0    # 40 hours runtime
#$ -l h_vmem=7.5G      # 11G RAM per core
#$ -l gpu=2           # request 2 GPU
#$ -l gpu_type=ampere
#$ -N nsid_tc_0_eval
#$ -o /data/home/acw723/NeuralSampleID/hpc_out
#$ -m beas

module load python/3.10.7
source ../grafp_venv/bin/activate
python test_fp.py --query_lens=5,7,10,15,20 \
                  --text=tc17_full_ivfpq \
                  --test_dir=../datasets/sample_100/audio
