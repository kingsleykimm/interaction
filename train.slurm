#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -A cral
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80gb
#SBATCH --mem=327680
#SBATCH --ntasks=1
#SBATCH -o train_out.txt

module purge
module load cuda apptainer pytorch

nvidia-smi


# remember to change this.
/scratch/bjb3az/.conda/envs/habitat/bin/python train.py --dataset_path kingsleykim/habitat_videos_13_12_24_10_45_17 --model_path ./models/LLaVA-Video-7B/