# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# datatag = ""
datatag = "_euler"
dataset = "spine_ct-mri"
# dataset = "spine_mri-ct"
num_classes = 6

_base_ = [
    "../_base_/default_runtime.py",
    # DAFormer Network Architecture
    "../_base_/models/segformer_r101.py",
    # GTA->Cityscapes Data Loading
    f"../_base_/datasets/uda_{dataset}_256x256{datatag}.py",
    # Basic UDA Self-Training
    "../_base_/uda/dacs_colormix.py",
    # AdamW Optimizer
    "../_base_/schedules/adamw.py",
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    "../_base_/schedules/poly10warm_med.py",
]

burnin_global = 1000
burnin = 0
uda = dict(
    color_mix=dict(
        burnin_global=burnin_global,
        burnin=burnin,
        coloraug=False,
        auto_bcg=False,
    )
)

norm_net = dict(norm_activation="linear", layers=[1, 1])
# norm_net = dict(norm_activation="relu", layers=[1, 32, 1])

model = dict(
    decode_head=dict(num_classes=num_classes),
    norm_cfg=norm_net,
)

seed = 0
# Modifications to Basic UDA
class_temp = 0.1
per_image = False
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=None
        # rare_class_sampling=dict(
        #     min_pixels=4, class_temp=class_temp, min_crop_ratio=0.5, per_image=per_image
        # )
    ),
)
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-04,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        )
    ),
)

n_gpus = 1
runner = dict(type="IterBasedRunner", max_iters=10000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=1)
evaluation = dict(interval=200, metric="mDice")
# Meta Information for Result Analysis


exp = "basic"
name_dataset = f"{dataset}{datatag}"
name_architecture = "segformer_r101"
name_encoder = "ResNetV1c"
name_decoder = "SegFormerHead"
name_uda = "dacs"
name_opt = "adamw_6e-05_pmTrue_poly10warm_1x2_30k"
name = f"{dataset}{datatag}_{name_architecture}-burnin{burnin}-g{burnin_global}-tinto"
