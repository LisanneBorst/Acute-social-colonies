#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --job-name=NWB_create
#SBATCH --mem-per-cpu=80GB

module purge
module load Python/3.10.4-GCCcore-11.3.0
 
source $HOME/.envs/taini_colonies/bin/activate
echo "Start datetime"
date

python /scratch/p304163/all_drd2_analysis/taini_colonies/src/nwb_create_with_filtering.py >> logs/log_nwb_create.txt

#echo "Done with creating NWB files"
#echo "Running nwb_add_event_trace.py ..."

#python src/nwb_add_event_trace.py >> logs/log_nwb_add_event_trace.txt

#echo "Done with adding event trace"
#echo "Running nwb_add_spatial_information.py ..."
#
#python src/nwb_add_spatial_information.py >> logs/log_nwb_add_coordinates.txt

echo "End datetime"
date
echo "Done mate!"

deactivate
