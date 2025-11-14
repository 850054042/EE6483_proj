#!/bin/bash
#SBATCH --job-name=catdog
#SBATCH --gpus=6000ada:1
#SBATCH --time=1:00:00
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

module load Miniforge3

source activate

conda activate py311

python main.py --dataset cifar10