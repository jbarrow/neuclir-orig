#!/bin/bash
#SBATCH --job-name=letor
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

. activate neuclir

rm -rf /fs/clip-scratch/jdbarrow/neuclir/experiments/__EXP_NAME__/

allennlp train runs/__EXP_NAME__.json --include-package neuclir -s /fs/clip-scratch/jdbarrow/neuclir/experiments/__EXP_NAME__
