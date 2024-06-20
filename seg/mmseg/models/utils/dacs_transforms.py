# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn
from torch import nn, optim
import cv2
# from sklearn.linear_model import LinearRegression
from mmseg.models.utils.dacs_normalization import NormNet


def cosine_annealing(
    step: int, total_steps: int, lr_max: float, lr_min: float, warmup_steps: int = 0
) -> float:
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos((step - warmup_steps) /
                       (total_steps - warmup_steps) * np.pi)
        )

    return lr


class ClasswiseMultAugmenter:
    def __init__(
        self,
        n_classes,
        coef,
        bias,
        suppress_bg: bool = True,
        auto_bcg: bool = False,
        device: str = "cuda:0",
        kernel_size: int = 3,
        total_steps: int = 10000,
        warm_up_iters: int = 1000,
        extra_flip: bool = False,
        learning_rate: float = 0.0001
    ):
        self.n_classes = n_classes
        self.device = device
        self.source_mean = -torch.ones(n_classes, 1)
        self.target_mean = -torch.ones(n_classes, 1)
        self.target_min = -torch.ones(n_classes, 1)
        self.coef = torch.zeros(n_classes, 3)
        self.bcg_shift = 0
        self.extra_flip = extra_flip

        self.suppress_bg = suppress_bg
        self.auto_bcg = auto_bcg
        self.kernel_size = (kernel_size, kernel_size)

        self.learning_rate = learning_rate
        self.normalization_net = NormNet(
            coef, bias, norm_activation="linear", layers=[1, 1]
        ).to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.normalization_net.parameters(), lr=self.learning_rate, weight_decay=0
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / self.learning_rate,
                warmup_steps=warm_up_iters,
            ),
        )

    def _normalize(self, img_original, background_mask):
        img_original_gray = img_original[:, 0, :, :].unsqueeze(1)

        with torch.no_grad():
            img = self.normalization_net(img_original_gray).detach()

        if self.suppress_bg:
            img[background_mask] = img_original_gray[background_mask] 
            # + self.coef[0, 0]

        img = img.repeat(1, 3, 1, 1)

        return img

    def optimization_step(
        self, optimizer_step, img_original, img_segm_hist, gt_semantic_seg, means, stds, auto_bcg=None
    ):
        denorm_(img_original, means, stds)
        denorm_(img_segm_hist, means, stds)

        img_segm_hist_gray = img_segm_hist[:, 0, :, :].clone().unsqueeze(1)
        img_original_gray = img_original[:, 0, :, :].unsqueeze(1)

        if auto_bcg is None:
            foreground_mask = gt_semantic_seg > 0
            background_mask = gt_semantic_seg == 0
        else:
            foreground_mask = auto_bcg > 0
            background_mask = auto_bcg == 0

        if optimizer_step:
            self.optimizer.zero_grad()

            img_polished = self.normalization_net(img_original_gray)

            if self.suppress_bg:

                loss = self.criterion(
                    img_polished[foreground_mask],
                    img_segm_hist_gray[foreground_mask].to(
                        img_polished.device),
                )
            else:
                loss = self.criterion(
                    img_polished, img_segm_hist_gray.to(img_polished.device)
                )

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss_val = loss.item()

            min_tgt = img_segm_hist_gray[foreground_mask].min().item()

            del img_polished

        else:
            loss_val = np.nan

        with torch.no_grad():
            img = self.normalization_net(img_original_gray).detach()

        if self.suppress_bg:
            img[background_mask] = img_segm_hist_gray[background_mask]

            # img[background_mask] = img_original[:, 0, :, :].unsqueeze(1)[background_mask].mean().item()

        img = img.repeat(1, 3, 1, 1)

        renorm_(img_original, means, stds)
        renorm_(img_segm_hist, means, stds)
        renorm_(img, means, stds)

        return img, loss_val

    def update(self, source, target, mask_src, mask_tgt, weight_tgt, param):
        """
        Update estimations of mean intensities per class.
        """
        mean, std = param["mean"], param["std"]
        denorm_(source, mean, std)
        denorm_(target, mean, std)

        if self.auto_bcg:
            source_bcg = []
            for i in range(source.shape[0]):
                source_bcg.append(find_background(source[i], self.kernel_size))
            source_bcg = torch.stack(source_bcg).to(self.device)

            target_bcg = []
            for i in range(target.shape[0]):
                target_bcg.append(find_background(target[i], self.kernel_size))
            target_bcg = torch.stack(target_bcg).to(self.device)
        else:
            source_bcg = mask_src == 0

        if self.extra_flip:
            source_ = self._normalize(source, source_bcg)
        else:
            source_ = source

        c = 0
        for i in range(self.n_classes):
            if self.auto_bcg:
                source_mean = source_[:, c, :, :][
                    (mask_src.squeeze(1) == i) & (source_bcg != 0)
                ]
                target_mean = target[:, c, :, :][
                    (mask_tgt.squeeze(1) == i)
                    & (target_bcg != 0)
                    & (weight_tgt.squeeze(1) > 0)
                ]
            else:
                source_mean = source_[:, c, :, :][mask_src.squeeze(1) == i]
                target_mean = target[:, c, :, :][
                    (mask_tgt.squeeze(1) == i) & (weight_tgt.squeeze(1) > 0)
                ]

            if source_mean.shape[0] != 0:
                self.source_mean[i, c] = source_mean.mean().item()

            if target_mean.shape[0] != 0:
                self.target_mean[i, c] = target_mean.mean().item()
                self.target_min[i, c] = target_mean.min().item()

            if (self.source_mean[i, c] != -1) and (self.target_mean[i, c] != -1):
                self.coef[i, c] = self.target_mean[i, c] - \
                    self.source_mean[i, c]

        renorm_(source, mean, std)
        renorm_(target, mean, std)

    def color_mix(self, data, mask, mean, std):
        data_ = data.clone()

        denorm_(data_, mean, std)

        if self.auto_bcg:
            background_mask = []
            for i in range(data_.shape[0]):
                background_mask.append(find_background(data_[i], self.kernel_size))
            background_mask = torch.stack(background_mask).to(self.device)
        else:
            background_mask = mask == 0

        if self.extra_flip:
            data_ = self._normalize(data_, background_mask)

        for i in range(self.n_classes):
            for c in range(3):
                if self.auto_bcg:
                    data_[:, c, :, :][
                        (mask.squeeze(1) == i) & (background_mask != 0)
                    ] += self.coef[i, 0]
                else:
                    data_[:, c, :, :][mask.squeeze(1) == i] += self.coef[i, 0]

        renorm_(data_, mean, std)
        if self.auto_bcg:
            return data_, background_mask.unsqueeze(1)
        else:
            return data_, None


def find_background(img, kernel_size):
    img.clamp_(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    image = (img * 255).astype(np.uint8)

    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
            image.shape) == 3 else image
    )

    # Apply thresholding to find markers
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal using morphological closing operation
    kernel = np.ones(kernel_size, np.uint8)
    closing = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Background area determination
    sure_bg = cv2.dilate(closing, kernel, iterations=1)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply the watershed
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    means_per_marker = []
    for m in np.unique(markers):
        means_per_marker.append(img[markers == m].mean())

    foreground_id = np.argmax(means_per_marker)
    final_mask = np.zeros_like(markers)
    final_mask[markers == np.unique(markers)[foreground_id]] = 1

    return torch.Tensor(final_mask)


def strong_transform(param, data=None, target=None):
    assert (data is not None) or (target is not None)

    data, target = one_mix(mask=param["mix"], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param["color_jitter"],
        s=param["color_jitter_s"],
        p=param["color_jitter_p"],
        mean=param["mean"],
        std=param["std"],
        data=data,
        target=target,
    )
    data, target = gaussian_blur(blur=param["blur"], data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]["img_norm_cfg"]["mean"], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]["img_norm_cfg"]["std"], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def renorm(img, mean, std):
    return img.mul_(255.0).sub_(mean).div_(std)


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter_med(color_jitter, mean, std, data=None, target=None, s=0.25, p=0.1):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=0
                        )
                    )
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def color_jitter(color_jitter, mean, std, data=None, target=None, s=0.25, p=0.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s
                        )
                    )
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2])
                        - 0.5
                        + np.ceil(0.1 * data.shape[2]) % 2
                    )
                )
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3])
                        - 0.5
                        + np.ceil(0.1 * data.shape[3]) % 2
                    )
                )
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)
                    )
                )
                data = seq(data)
    return data, target


def get_class_masks(labels, ignore_index=0):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        classes = classes[classes != ignore_index]
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False
        )

        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))

    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(
        label, classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] + (1 - stackedMask0) * target[1]).unsqueeze(
            0
        )
    return data, target
