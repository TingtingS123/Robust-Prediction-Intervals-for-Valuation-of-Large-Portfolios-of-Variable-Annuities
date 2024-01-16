#!/bin/bash
#SBATCH --job-name=bootstrap  # create a name for your job
#SBATCH --nodes=4            # node count
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=research
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module purge
module load prun
module load gnu12
module load openmpi4
module load py3-mpi4py
module load py3-numpy
source ~/mypython/mypython/bin/activate
module load cmake

mpiexec -n 10 python3 bootstrap_version1.py
                                           

