#!/bin/bash
#SBATCH -t 6:00:00
#SBATCH -A cral
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100_80gb
#SBATCH --ntasks=1
#SBATCH -o dump.txt

module purge
module load cuda apptainer pytorch

/scratch/bjb3az/.conda/envs/habitat/bin/python labelling.py