# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn
from torch import nn, optim
import cv2


class ClasswiseMultAugmenter:
    def __init__(
        self,
        n_classes,
        suppress_bg: bool = True,
        auto_bcg: bool = False,
        device: str = "cuda:0",
        kernel_size: int = 3
    ):
        self.n_classes = n_classes
        self.device = device
        self.source_mean = -torch.ones(n_classes, 1)
        self.target_mean = -torch.ones(n_classes, 1)
        self.coef = torch.zeros(n_classes, 3)

        self.suppress_bg = suppress_bg
        self.auto_bcg = auto_bcg
        self.kernel_size = (kernel_size, kernel_size)

    def update(self, source, target, mask_src, mask_tgt, weight_tgt, param):
        mean, std = param["mean"], param["std"]
        denorm_(source, mean, std)
        denorm_(target, mean, std)

        if self.auto_bcg:
            source_bcg = []
            for i in range(source.shape[0]):
                source_bcg.append(self.find_background(source[i]))
            source_bcg = torch.stack(source_bcg).to(self.device)

            target_bcg = []
            for i in range(target.shape[0]):
                target_bcg.append(self.find_background(target[i]))
            target_bcg = torch.stack(target_bcg).to(self.device)

        c = 0
        for i in range(self.n_classes):
            if self.auto_bcg:
                source_mean = source[:, c, :, :][
                    (mask_src.squeeze(1) == i) & (source_bcg != 0)
                ]
                target_mean = target[:, c, :, :][
                    (mask_tgt.squeeze(1) == i)
                    & (target_bcg != 0)
                    & (weight_tgt.squeeze(1) > 0)
                ]
            else:
                source_mean = source[:, c, :, :][mask_src.squeeze(1) == i]
                target_mean = target[:, c, :, :][
                    (mask_tgt.squeeze(1) == i) & (weight_tgt.squeeze(1) > 0)
                ]

            if source_mean.shape[0] != 0:
                self.source_mean[i, c] = source_mean.mean().item()

            if target_mean.shape[0] != 0:
                self.target_mean[i, c] = target_mean.mean().item()

            if (self.source_mean[i, c] != -1) and (self.target_mean[i, c] != -1):
                self.coef[i, c] = self.target_mean[i, c] - self.source_mean[i, c]

        renorm_(source, mean, std)
        renorm_(target, mean, std)

    def color_mix(self, data, mask, mean, std):
        data_ = data.clone()

        # denorm_(data_, mean, std)

        if self.auto_bcg:
            background_mask = []
            for i in range(data_.shape[0]):
                background_mask.append(
                    self.find_background(data_[i])
                )
            background_mask = torch.stack(background_mask).to(self.device)

        for i in range(self.n_classes):
            for c in range(3):
                # old_min = data_[:, c, :, :][mask.squeeze(1) != 0].min()
                if self.auto_bcg:
                    data_[:, c, :, :][
                        (mask.squeeze(1) == i) & (background_mask != 0)
                    ] += self.coef[i, 0]
                else:
                    data_[:, c, :, :][mask.squeeze(1) == i] += self.coef[i, 0]

                # new_min = data_[:, c, :, :][mask.squeeze(1) != 0].min()

                # data_[:, c, :, :][mask.squeeze(1) != 0] += old_min - new_min

        # min_val = data_.min()
        # data_ -= min_val
        # max_val = data_.max()
        # if max_val != 0:
        #     data_ /= max_val

        # renorm_(data_, mean, std)
        if self.auto_bcg:
            return data_[:, 0, :, :].unsqueeze(1), background_mask.unsqueeze(1)
        else:
            return data_[:, 0, :, :].unsqueeze(1), None

    def find_background(self, img):
        img.clamp_(0, 1)
        img = img.permute(1, 2, 0).cpu().numpy()
        image = (img * 255).astype(np.uint8)

        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )

        # Apply thresholding to find markers
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Noise removal using morphological closing operation
        kernel = np.ones(self.kernel_size, np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Background area determination
        sure_bg = cv2.dilate(closing, kernel, iterations=1)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

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
    label, classes = torch.broadcast_tensors(label, classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] + (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] + (1 - stackedMask0) * target[1]).unsqueeze(
            0
        )
    return data, target
