#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks 25
#SBATCH --output arlTest6.out
#SBATCH --qos himem

module load matlab
module load slurm
setenv HOME /projects/jmf/arl

matlab -nodesktop -nosplash -r "run"
