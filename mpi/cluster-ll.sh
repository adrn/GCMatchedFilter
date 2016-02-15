#!/bin/sh

# Directives
#PBS -N globber
#PBS -W group_list=yetiastro
#PBS -l nodes=1:ppn=1,walltime=00:30:00,mem=4gb
#PBS -V
#PBS -t 1725,8761,8850

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

module load openmpi/1.6.5-no-ib

# print date and time to file
date

cd /vega/astro/users/amp2217/projects/globber/

source activate globber

# New run
python scripts/compute-cmd-likelihoods -f data/ngc5897/PS1_stars_pv3_dered.npy -n 1000 -i $PBS_ARRAYID

date

#End of script
