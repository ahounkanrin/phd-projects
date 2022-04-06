#!/bin/sh

#SBATCH --account=gpuo
#SBATCH --partition=gpuo
#SBATCH --nodes=1 --ntasks=10
#SBATCH --time=6-0
#SBATCH --gres=gpu:kepler:1
#SBATCH --job-name="Pos5000"
#SBATCH --mail-user=hnkmah001@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

# The cluster is configured primarily for OpenMPI and PMI. Use srun to launch parallel jobs if your code is parallel aware.
# To protect the cluster from code that uses shared memory and grabs all available cores the cluster has the following 
# environment variable set by default: OMP_NUM_THREADS=1
# If you feel compelled to use OMP then uncomment the following line:
# export OMP_NUM_THREADS=$SLURM_NTASKS

# NB, for more information read https://computing.llnl.gov/linux/slurm/sbatch.html

# Use module to gain easy access to software, typing module avail lists all packages.

# Your science stuff goes here...
#export CUDA_VISIBLE_DEVICES=$(ncvd)
#module load software/TensorFlow-A100-GPU
module load software/TensorFlow-2x-GPU
python train.py --batch_size=32


