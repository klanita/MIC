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
        "type": "fda",
        "L": 0.09,
        "coloraug": True,        
        "color_jitter_s": 0.1,
        "color_jitter_p": 0.25,
        "gaussian_blur": False,
        "blur": 0.1
    },
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
)
use_ddp_wrapper = True