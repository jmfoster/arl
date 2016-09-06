#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --output hello-world.out
#SBATCH --qos janus-debug

echo "Running on $(hostname --fqdn): echo 'Hello, world!'