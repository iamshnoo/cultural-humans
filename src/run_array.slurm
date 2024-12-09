#!/bin/bash

#SBATCH --job-name=editeval1
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=40GB
#SBATCH --time=0-02:00:00
#SBATCH --array=1-20

readarray -t ARGS < args.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}
echo $ARGUMENTS
#python edit.py $ARGUMENTS
python evals.py $ARGUMENTS
