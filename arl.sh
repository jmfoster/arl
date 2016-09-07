#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --output arl.out
#SBATCH --qos janus

module load matlab
module load slurm
setenv HOME /projects/jmf/arl

matlab -nodesktop -nosplash -r "run"