#!/bin/bash

#SBATCH --output arl.out
#SBATCH --qos janus-debug

module load matlab
module load slurm
cd /projects/jmf/arl

matlab -nodesktop -nosplash -r "run"