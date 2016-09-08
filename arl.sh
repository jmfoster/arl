#!/bin/bash

#SBATCH --ntasks 40
#SBATCH --output arl.out
#SBATCH --qos himem

module load matlab
module load slurm
setenv HOME /projects/jmf/arl

matlab -nodesktop -nosplash -r "run"