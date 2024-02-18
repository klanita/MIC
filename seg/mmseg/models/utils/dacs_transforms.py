# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn
# from mmseg.models.utils.dacs_normalization import NormNet
from torch import nn, optim
from sklearn.linear_model import LinearRegression

class ClasswiseMultAugmenter:
    def __init__(self, n_classes, norm_type: str, suppress_bg: bool=True, device: str="cuda:0"):
        self.n_classes = n_classes
        self.device = device
        self.source_mean = -torch.ones(n_classes, 1)
        self.target_mean = -torch.ones(n_classes, 1)
        self.delta = torch.full((n_classes,), float('nan'))
        # torch.zeros(n_classes)

        self.suppress_bg = suppress_bg

        # self.learning_rate = 6e-05
        self.learning_rate = 0.01
        self.normalization_net = nn.Conv2d(1, 1, kernel_size=1, bias=True).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.normalization_net.parameters(), lr=self.learning_rate, weight_decay=0)

    def optimization_step(self, img_original, img_segm_hist, gt_semantic_seg):               
        self.optimizer.zero_grad()
        # print(img_original.device, img_segm_hist.device, gt_semantic_seg.device)
        # print(next(self.normalization_net.parameters()).device)
        # quit()

        img_polished = self.normalization_net(img_original[:, 0, :, :].unsqueeze(1)) 
        
        if self.suppress_bg:
             ## automatically detect background value
            # background_val = img_original[0, 0, 0, 0].item()
            # foreground_mask = img_original[:, 0, :, :].unsqueeze(1) > 0
            # background_mask = img_original[:, 0, :, :].unsqueeze(1) == background_val
            
            foreground_mask = gt_semantic_seg > 0
            background_mask = gt_semantic_seg == 0

            loss = self.criterion(img_polished[foreground_mask], img_segm_hist[foreground_mask].to(img_polished.device))
        else:
            loss = self.criterion(img_polished, img_segm_hist.to(img_polished.device))       

        loss.backward()
        self.optimizer.step()

        min_hist, max_hist = img_segm_hist[foreground_mask].min().item(), img_segm_hist[foreground_mask].max().item()

        # img = img_polished.detach()
        del img_polished
        with torch.no_grad():
            img = self.normalization_net(img_original[:, 0, :, :].unsqueeze(1)).detach()
            
        # img[foreground_mask] += max_hist - img[foreground_mask].max().item()

        if self.suppress_bg:
            img[background_mask] = img_segm_hist[background_mask]
            # img[background_mask] = img_original[:, 0, :, :].unsqueeze(1)[background_mask].mean().item()

        img = img.repeat(1, 3, 1, 1)

        return img, loss.item()
        
    def update(self, source, target, mask_src, mask_tgt, weight_tgt, param, alpha=0.95):
        mean, std = param["mean"], param["std"]
        denorm_(source, mean, std)
        denorm_(target, mean, std)

        c = 0
        for i in range(self.n_classes):            
            source_mean = source[:, c, :, :][mask_src.squeeze(1) == i]
            target_mean = target[:, c, :, :][(mask_tgt.squeeze(1) == i) & (weight_tgt.squeeze(1) > 0)]

            if (source_mean.shape[0] != 0):
                if self.source_mean[i, c] == -1:
                    self.source_mean[i, c] = source_mean.mean().item()
                else:
                    self.source_mean[i, c] = alpha*source_mean.mean().item() + (1-alpha)*self.source_mean[i, c]

            if (target_mean.shape[0] != 0):
                if self.target_mean[i, c] == -1:
                    self.target_mean[i, c] = target_mean.mean().item()
                else:
                    self.target_mean[i, c] = alpha*target_mean.mean().item() + (1-alpha)*self.target_mean[i, c]

            if (self.source_mean[i, c] != -1) and (self.target_mean[i, c] != -1):
                self.delta[i] = self.target_mean[i, c] - self.source_mean[i, c]
                    
        renorm_(source, mean, std)
        renorm_(target, mean, std)

    def color_mix(self, data, mask, mean, std):
        has_nan = torch.isnan(self.delta[1:]).all()

        if has_nan:            
            return None
        else:
            data_before_ = data.clone()
            data_ = data.clone()

            denorm_(data_, mean, std)
            denorm_(data_before_, mean, std)

            for cl in range(self.n_classes):
                for c in range(3):
                    # old_min = data_[:, c, :, :][mask.squeeze(1) != 0].min()
                    if not torch.isnan(self.delta[cl]):
                        data_[:, c, :, :][mask.squeeze(1) == cl] += self.delta[cl]

                    # new_min = data_[:, c, :, :][mask.squeeze(1) != 0].min()

                    # data_[:, c, :, :][mask.squeeze(1) != 0] += old_min - new_min
                
            for i in range(data_.shape[0]):
                # min_val = data_.min()
                # data_[i] -= min_val
                # max_val = data_[i].max()
                # if max_val != 0:
                #     data_[i] /= max_val

                lin_pred = self._linear_match_cost(data_before_[i, 0].cpu().numpy(), data_[i, 0].cpu().numpy(), mask[i].cpu().numpy())
                for c in range(3):
                    data_[i, c] = lin_pred

                # min_val = data_.min()
                # data_[i] -= min_val
                # max_val = data_[i].max()
                # if max_val != 0:
                #     data_[i] /= max_val

            renorm_(data_, mean, std)

            return data_[:, 0, :, :].unsqueeze(1)

    def _linear_match_cost(self, source, template, mask):
        h, w = source.shape
        source_vec = source.reshape(-1)
        template_vec = template.reshape(-1)
        mask_vec = mask.reshape(-1)
        # DecisionTreeRegressor(max_depth=10)
        # MLPRegressor(hidden_layer_sizes=(256,256,256))
        reg = LinearRegression().fit(source_vec[mask_vec != 0].reshape(-1, 1), template_vec[mask_vec != 0])
        source_new = np.zeros(h*w)
        source_new[mask_vec != 0] = reg.predict(source_vec[mask_vec != 0].reshape(-1, 1))
        source_new[mask_vec == 0] = template_vec[mask_vec == 0]

        # print('source', source_vec.min(), source_vec.max())
        # print('template', template_vec.min(), template_vec.max())
        # print('source_new', source_new.min(), source_new.max())
        source_new = torch.Tensor(source_new.reshape(h, w)).to(self.device)
        # source_new.clamp(0, 1)
        return source_new
    
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
