#!/bin/bash
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --job-name=form_repr
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --mail-user=labash.b@northeastern.edu
#SBATCH --mail-type=BEGIN
sleep infinity