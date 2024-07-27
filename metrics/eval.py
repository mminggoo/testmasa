from typing import Any, Callable, Dict, List, Optional, Union
import PIL.Image
from PIL import Image
import numpy as np
import torch
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

device = 'cuda' if torch.cuda.is_available else 'cpu'


clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
clip_score_fn_large = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")

lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex') # alex vgg squeeze
loss_fn_vgg = LearnedPerceptualImagePatchSimilarity(net_type='vgg') # alex vgg squeeze

ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)

def calculate_clip_score(images: List[PIL.Image.Image], prompts: List[str]):

    base_scores = []
    large_scores = []

    for image, prompt in zip(images, prompts):
        image = np.array(image)
        img_tensor = torch.tensor(image).permute(2,0,1)
        
        score_base = clip_score_fn(img_tensor, prompt)
        score_base = score_base.cpu().item()
        score_large = clip_score_fn_large(img_tensor, prompt)
        score_large = score_large.cpu().item()

        base_scores.append(score_base)
        large_scores.append(score_large)
    return np.mean(base_scores), np.mean(large_scores)

def calculate_lpips(images0: List[PIL.Image.Image], images1: List[PIL.Image.Image], mask_preds=None, mask_gts=None):

    squeeze_scores = []
    vgg_scores = []
    alex_scores = []
    for i in range(len(images0)):
        img_pred = images0[i]
        img_gt = images1[i]

        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255

        if mask_preds is not None:
            mask_pred = mask_preds[i]
            mask_pred = np.array(mask_pred).astype(np.float32)
            mask_pred = np.where(mask_pred != 0, 1.0, 0.0).astype(np.float32)
            mask_pred = np.stack([mask_pred] * 3, axis=-1) if mask_pred.shape[-1] != 3 else mask_pred
            img_pred = img_pred * mask_pred
        if mask_gts is not None:
            mask_gt = mask_gts[i]
            mask_gt = np.array(mask_gt).astype(np.float32)
            mask_gt = np.where(mask_gt != 0, 1.0, 0.0).astype(np.float32)
            mask_gt = np.stack([mask_gt] * 3, axis=-1) if mask_gt.shape[-1] != 3 else mask_gt
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2,0,1).unsqueeze(0)
        img_gt_tensor = torch.tensor(img_gt).permute(2,0,1).unsqueeze(0)

        squeeze_lpips_score = lpips_metric_calculator(img_pred_tensor*2-1,img_gt_tensor*2-1)
        squeeze_lpips_score = squeeze_lpips_score.cpu().item()
        alex_lpips_score = loss_fn_alex(img_pred_tensor*2-1,img_gt_tensor*2-1)
        alex_lpips_score = alex_lpips_score.cpu().item()
        vgg_lpips_score = loss_fn_vgg(img_pred_tensor*2-1,img_gt_tensor*2-1)
        vgg_lpips_score = vgg_lpips_score.cpu().item()

        squeeze_scores.append(squeeze_lpips_score)
        vgg_scores.append(vgg_lpips_score)
        alex_scores.append(alex_lpips_score)
    return np.mean(squeeze_scores), np.mean(vgg_scores), np.mean(alex_scores)

def calculate_ssim(images0, images1, mask_preds=None, mask_gts=None):
    ssim = []

    for i in range(len(images0)):
        img_pred = images0[i]
        img_gt = images1[i]

        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255

        if mask_preds is not None:
            mask_pred = mask_preds[i]
            mask_pred = np.array(mask_pred).astype(np.float32)
            mask_pred = np.where(mask_pred != 0, 1.0, 0.0).astype(np.float32)
            mask_pred = np.stack([mask_pred] * 3, axis=-1) if mask_pred.shape[-1] != 3 else mask_pred
            img_pred = img_pred * mask_pred
        if mask_gts is not None:
            mask_gt = mask_gts[i]
            mask_gt = np.array(mask_gt).astype(np.float32)
            mask_gt = np.where(mask_gt != 0, 1.0, 0.0).astype(np.float32)
            mask_gt = np.stack([mask_gt] * 3, axis=-1) if mask_gt.shape[-1] != 3 else mask_gt
            img_gt = img_gt * mask_gt
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(device)

        score = ssim_metric_calculator(img_pred_tensor,img_gt_tensor)
        score = score.cpu().item()
    
        ssim.append(score)

    return np.mean(ssim)

def calculate_psnr(images0, images1, mask_preds=None, mask_gts=None):
    psnr = []

    for i in range(len(images0)):
        img_pred = images0[i]
        img_gt = images1[i]
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255

        if mask_preds is not None:
            mask_pred = mask_preds[i]
            mask_pred = np.array(mask_pred).astype(np.float32)
            mask_pred = np.where(mask_pred != 0, 1.0, 0.0).astype(np.float32)
            mask_pred = np.stack([mask_pred] * 3, axis=-1) if mask_pred.shape[-1] != 3 else mask_pred
            img_pred = img_pred * mask_pred
        if mask_gts is not None:
            mask_gt = mask_gts[i]
            mask_gt = np.array(mask_gt).astype(np.float32)
            mask_gt = np.where(mask_gt != 0, 1.0, 0.0).astype(np.float32)
            mask_gt = np.stack([mask_gt] * 3, axis=-1) if mask_gt.shape[-1] != 3 else mask_gt
            img_gt = img_gt * mask_gt
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(device)
        
        score = psnr_metric_calculator(img_pred_tensor,img_gt_tensor)
        score = score.cpu().item()
    
        psnr.append(score)

    return np.mean(psnr)