#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --job-name=NWB_create
#SBATCH --mem-per-cpu=8GB

module purge
module load Python/3.10.4-GCCcore-11.3.0
 
source $HOME/.envs/taini_colonies/bin/activate

python /scratch/p304163/all_drd2_analysis/taini_colonies/src/create_edf_metadata.py

deactivate
