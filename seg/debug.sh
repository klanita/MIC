#!/bin/bash

# srun --time 480 --account=staff --partition=gpu.debug --mem=50G --gres=gpu:1 --pty bash -i 
# srun --time 480 --partition=gpu.debug --gres=gpu:1 --mem=80G --pty bash -i

#SBATCH  --output=./LOGS/%j.out
#SBATCH  --error=./LOGS/%j.out
#SBATCH  --gpus=1
#SBATCH  --mem-per-cpu=5G
##SBATCH  --mem-per-cpu=10G
#SBATCH  --ntasks=1
#SBATCH  --cpus-per-task=2
#SBATCH  --time=1-0

# srun --time=1-0 --gpus=1 --mem-per-cpu=5G --pty bash -i

# source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
# conda activate mic

WANDB_MODE=disabled
WORKPATH=/usr/bmicnas02/data-biwi-01/klanna_data/results/MIC/

# python run_experiments.py --config configs/brain/debug.py

# python run_experiments.py --config configs/brain/baseline_cnn.py

# python run_experiments.py --config configs/brain/segformer_rcs.py

# python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py

# python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_contrastive.py 

# python run_experiments.py --config configs/hrda/gtaHR2csHR_hrda.py

# python run_experiments.py --config configs/whitematter/daformer_mic.py

python run_experiments.py --config configs/wmh/segformer.py


# python run_experiments.py --config configs/brain/daformer_mic_colormix_source.py

# python run_experiments.py --config configs/brain/segformer_colormix_source.py

# python run_experiments.py --config configs/wmh/segformer_colormix_tinto.py

# python run_experiments.py --config configs/spine/segformer_src.py

# python run_experiments.py --config configs/hcp/segformer_fda.py
