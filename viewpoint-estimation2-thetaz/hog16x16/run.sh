#!/bin/sh

# This example submission script contains several important directives, please examine it thoroughly

# The line below indicates which accounting group to log your job against
##SBATCH --account=gpuo --partition=gpuo --nodes=1 --ntasks=12 --time=5-0 --job-name="hog16" --gres gpu:kepler:1

#SBATCH --account=eleceng --partition=ada --nodes=1 --ntasks=40 --time=7-0 --job-name="hog16" 

#Modify the lines below for email alerts. Valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL 
#SBATCH --mail-user=hnkmah001@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

# The cluster is configured primarily for OpenMPI and PMI. Use srun to launch parallel jobs if your code is parallel aware.
# To protect the cluster from code that uses shared memory and grabs all available cores the cluster has the following 
# environment variable set by default: OMP_NUM_THREADS=1
# If you feel compelled to use OMP then uncomment the following line:
# export OMP_NUM_THREADS=$SLURM_NTASKS

# NB, for more information read https://computing.llnl.gov/linux/slurm/sbatch.html

# Use module to gain easy access to software, typing module avail lists all packages.
# Example:
# module load python/anaconda-python-2.7

# Your science stuff goes here...
module load software/TensorFlow-2x-GPU
python train.py 




