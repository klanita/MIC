# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add MIC options
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Baseline UDA
_base_ = ['dacs.py']
uda = dict(
    alpha=0.999,
    color_mix={
        "type": "source",
        "burnin_global": 0,
        "freq": 1.0,
        "weight": 1.0,
        "bias": 0.0,
        "suppress_bg": True,
        "norm_type": "linear",
        "burnin": -1,
        "burninthresh": 1.0,
        "coloraug": True,        
        "color_jitter_s": 0.1,
        "color_jitter_p": 0.25,
        "gaussian_blur": False,
        "blur": 0.1,
        "auto_bcg": False
    },
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    debug_img_interval=100,
)
use_ddp_wrapper = True