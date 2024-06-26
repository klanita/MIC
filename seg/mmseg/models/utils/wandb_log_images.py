
import wandb
from torchvision.utils import make_grid
import torch
from PIL import Image
import numpy as np
# dict_keys(['Img 0 Scale 0.5', 'Img 1 Scale 1', 'High Res', 'High Res Inserted', 'Low Res', 'Fused', 'Attention', 'Cropped GT', 'GT'])
# 
def WandbLogImages(seg_debug, max_images: int = 4):
                #    outputs, imgs, targets, name: str, num_classes: int):
                # the following code logs images with targets, predictions and losses
        # to wandb. (the values are attached as captions)
        # self.logger: WandbLogger

        if len(seg_debug['Source'].keys()):
            print(seg_debug['Source'].keys())
            if 'Seg. Pred.' in seg_debug['Source']:
                num_classes = seg_debug['Source']['Seg. Pred.'].shape[1]
            
                class_labels = dict(zip(range(num_classes), [str(i+1) for i in range(num_classes)]))
            
                n_images = min(len(seg_debug['Source']['Image']), max_images)

                for domain in ['Source', 'Mix']:       
                    if domain in seg_debug:
                        for i in range(n_images):
                            img_to_plot = seg_debug[domain]['Image'][i].permute(1, 2, 0).cpu().numpy()
                            prediction = seg_debug[domain]['Seg. Pred.'].argmax(1)[i]
                            target = seg_debug[domain]['Seg. GT'][i]
                            wandb.log(
                                {f'Debug-{domain}': wandb.Image(img_to_plot, 
                                    masks={
                                        "predictions" : {
                                            "mask_data" : prediction,
                                            "class_labels": class_labels
                                        },
                                        "ground_truth" : {
                                            "mask_data" : target,
                                            "class_labels": class_labels
                                        }
                                    }
                                )})

                domain = 'Target'
                if domain in seg_debug:
                    for i in range(n_images):
                        img_to_plot = seg_debug[domain]['Image'][i].permute(1, 2, 0).cpu().numpy()
                        prediction = seg_debug[domain]['Pred'].argmax(1)[i]
                        wandb.log(
                            {f'Debug-{domain}': wandb.Image(img_to_plot, 
                                masks={
                                    "predictions" : {
                                        "mask_data" : prediction,
                                        "class_labels": class_labels
                                    }
                                }
                            )})
                    
PALETTE = [[153, 153, 153], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70]]

def convert_3d(masks, PALETTE):
    masks_3d = np.zeros([3, 256, 256])
    for i in range(256):
        for j in range(256):
            l = int(masks[i, j])
            masks_3d[:, i, j] = np.array(PALETTE[l])
    return masks_3d

def WandbLogPredictions(results, gt_seg_maps, PALETTE, max_images=16):
    max_images = min(max_images, len(results))
    idx = list(np.random.choice(len(results), max_images))

    masks = make_grid([torch.Tensor(convert_3d(results[i], PALETTE)) for i in idx] \
                        + [torch.Tensor(convert_3d(gt_seg_maps[i], PALETTE))for i in idx], 
                        nrow=max_images).permute(1, 2, 0).numpy()

    results_pil = Image.fromarray(np.uint8(masks)).convert('RGB')
    results_pil.save(f"tmp.png","PNG")
    
    wandb.log({"Validation": [wandb.Image(results_pil)]})