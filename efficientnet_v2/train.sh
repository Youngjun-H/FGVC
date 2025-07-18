#!/bin/bash
#SBATCH --job-name=infer_test
#SBATCH --nodelist=hpe162
#SBATCH --gpus=8
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="LLM tokenizer 실험"

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "실험 내용"
echo "이미지 증강 기법 변경 1차" 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

# Huggingface cache directory
export HF_HOME=./cache/hf

# Kagglehub cache directory
export KAGGLEHUB_CACHE=./cache/kagglehub

# torch cache directory
export TORCH_HOME=./cache/torch

srun python src/train.py