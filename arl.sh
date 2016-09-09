#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks 40
#SBATCH --output arl4.out
#SBATCH --qos himem
#SBATCH --time 8:0:0

module load matlab
module load slurm
setenv HOME /projects/jmf/arl

matlab -nodesktop -nosplash -r "run"