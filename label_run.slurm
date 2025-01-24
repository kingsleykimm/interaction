#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -A cral
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=163840
#SBATCH --ntasks=1
#SBATCH -o other_label_run.txt

module load anaconda
module load cuda

nvcc --version
nvidia-smi

conda activate habitat

/scratch/bjb3az/.conda/envs/habitat/bin/python labelling.py
# /scratch/bjb3az/.conda/envs/habitat/bin/python upload_dataset.py --dataset_path good_data/ --num_frames 12