#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --job-name=NWB_coords
#SBATCH --mem-per-cpu=80GB

module purge
module load Python/3.10.4-GCCcore-11.3.0
 
source $HOME/.envs/taini_colonies/bin/activate
echo "Running nwb_add_spatial_information.py ..."
date

python taini_colonies/src/nwb_add_spatial_information.py >> logs/log_nwb_add_coordinates.txt

echo "Done"
date
deactivate
