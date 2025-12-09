#!/bin/sh

#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --time 8:00:00
#SBATCH --array=0-58319
#SBATCH --mail-user=yarno.dejaeger@ugent.be
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

cd $VSC_DATA_VO_USER

module load Python/3.12.3

source venvs/gcn_env/bin/activate

python machine_learning/Python/VAE_inner_folds_performance.py $PBS_ARRAYID