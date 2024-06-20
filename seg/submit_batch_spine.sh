#!/bin/bash
# sbatch ./euler_submit.sh --config configs/spine/segformer_fda.py
# sbatch ./euler_submit.sh --config configs/spine/segformer_fda.py

# sbatch ./euler_submit.sh --config configs/spine/segformer_gan.py
# sbatch ./euler_submit.sh --config configs/spine/segformer_gan.py

# sbatch ./euler_submit.sh --config configs/spine/segformer.py
# sbatch ./euler_submit.sh --config configs/spine/segformer.py

# sbatch ./euler_submit.sh --config configs/spine/segformer_src.py

# sbatch ./euler_submit.sh --config configs/spine/segformer_colormix_extra.py


sbatch ./euler_submit.sh --config configs/spine/segformer_colormix.py
sbatch ./euler_submit.sh --config configs/spine/segformer_colormix_1.py

sbatch ./euler_submit.sh --config configs/spine/segformer_colormix_flip.py
sbatch ./euler_submit.sh --config configs/spine/segformer_colormix_flip_1.py

sbatch ./euler_submit.sh --config configs/spine/segformer_colormix_bcg.py
sbatch ./euler_submit.sh --config configs/spine/segformer_colormix_bcg_1.py