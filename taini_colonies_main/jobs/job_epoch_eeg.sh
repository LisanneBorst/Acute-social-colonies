#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=epoch_eeg
#SBATCH --mem-per-cpu=120GB

module purge
module load Python/3.10.4-GCCcore-11.3.0
 
source $HOME/.envs/taini_colonies/bin/activate
echo "Start datetime"
date

python /scratch/p304163/all_drd2_analysis/taini_colonies/src/mass_epoch_eeg.py >> logs/log_epoch_eeg.txt


echo "End datetime"
date
echo "Done mate!"

deactivate
