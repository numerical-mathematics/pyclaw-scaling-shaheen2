#!/bin/bash

#SBATCH -N 1
#SBATCH -J 4_acoustics
#SBATCH -o 4_acoustics.out
#SBATCH -A k1069
#SBATCH -t 03:00:00

echo job running on...
hostname

module load python/.pyclaw_64bits_493

export OMP_NUM_THREADS=1
export PYTHONPATH=/project/k1069/lib/python2.7/site-packages:$PYTHONPATH

srun --ntasks=4 --cpus-per-task=1 --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --ntasks-per-core=1 --cpu_bind=cores python ../acoustics_3d_scaling.py -s strong -x 256
