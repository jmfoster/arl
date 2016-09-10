#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --output arl7.out
#SBATCH --qos himem

module load matlab
module load slurm
setenv HOME /projects/jmf/arl

matlab -nodesktop -nosplash -r "run"