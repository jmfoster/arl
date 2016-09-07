#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --output arl.out
#SBATCH --qos janus-debug

module load matlab
module load slurm
cd /projects/jmf/arl

addpath("Cache")
matlab run.m