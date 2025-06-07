#!/bin/bash
#SBATCH --job-name=LLM_notebook
#SBATCH --nodelist=hpe161
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="Qwen2.5 TEST"
#SBATCH --output=notebook_%A.log

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

hostname -I
# Huggingface cache directory
# export HF_HOME=./cache/hf
export HF_HOME=/purestorage/AILAB/AI_2/yjhwang/work/vlm2/cache/hf

# Kagglehub cache directory
# export KAGGLEHUB_CACHE=./cache/kagglehub
export KAGGLEHUB_CACHE=/purestorage/AILAB/AI_2/yjhwang/work/vlm2/cache/kagglehub
export TORCH_HOME=/purestorage/AILAB/AI_2/yjhwang/work/vlm2/cache/torch

srun jupyter notebook --ip 0.0.0.0