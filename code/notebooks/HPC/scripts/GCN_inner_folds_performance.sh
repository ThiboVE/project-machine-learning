#!/bin/sh

#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --time 6:00:00
#SBATCH --array=0-1214
#SBATCH --mail-user=yarno.dejaeger@ugent.be
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

cd $VSC_DATA

module load Python/3.12.3

source gcn_env/bin/activate

python Python/GCN_inner_folds_performance.py $PBS_ARRAYID