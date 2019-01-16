#!/bin/bash
#SBATCH --job-name=letor
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --mem=16GB
#SBATCH --time=02:00:00

allennp train __EXPNAME__.json --include-package neuclir -s ../experiments/__EXPNAME__
