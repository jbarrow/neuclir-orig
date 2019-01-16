#!/bin/bash
#SBATCH --job-name=letor
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --mem=16GB
#SBATCH --time=02:00:00

. activate neuclir

allennlp train __EXPNAME__.json --include-package neuclir -s ../experiments/__EXPNAME__
