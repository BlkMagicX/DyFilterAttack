import torch
import numpy as np


def compute_direction(alpha_o, alpha_t):
    return torch.sign(alpha_t - alpha_o)  # (B,N,C,1,1)


def compute_basic_weights(grad_orig, grad_target):
    """
    Channel-wise weights W_k^± for each detection.

    Args
    ----
    grad_orig   : Tensor, shape (B, N, C, H, W)
    grad_target : Tensor, shape (B, N, C, H, W)

    Returns
    -------
    alpha_t : Tensor, shape (B, N, C, 1, 1)
            suppress original class
    alpha_o : Tensor, shape (B, N, C, 1, 1)
            promote target class
    """
    # Global-average-pool over spatial dims
    alpha_o = grad_orig.mean(dim=(3, 4), keepdim=True)  # (B,N,C,1,1)
    alpha_t = grad_target.mean(dim=(3, 4), keepdim=True)  # (B,N,C,1,1)

    return alpha_o, alpha_t


def compute_spatial_mask(grad_orig):
    """
    Spatial map S_k ∈ [0,1] for each detection.

    Args
    ----
    grad_orig : Tensor, shape (B, N, C, H, W)

    Returns
    -------
    spatial_mask : Tensor, shape (B, N, C, H, W)
    """
    grad_abs = grad_orig.abs()  # (B,N,C,H,W)
    # Per-channel global max |g|  →  (B,N,C,1,1)
    grad_max = grad_abs.amax(dim=(3, 4), keepdim=True)

    spatial_mask = grad_abs / (grad_max + 1e-8)  # normalised 0-1 (B,N,C,H,W)

    return spatial_mask


def compute_full_weight(alpha_t, alpha_o, direction):

    weight_p = torch.where(direction > 0, alpha_t.abs(), alpha_o.abs())
    weight_n = torch.where(direction < 0, -alpha_o.abs(), -alpha_t.abs())

    return weight_p, weight_n


def compute_decouple_mask(weight_p: torch.Tensor, weight_n: torch.Tensor, spatial_mask: torch.Tensor):
    """
    Element-wise masks M_k^± = W_k^± ⊗ S_k.

    Args
    ----
    weight_p : Tensor, (B, N, C, 1, 1)
    weight_n : Tensor, (B, N, C, 1, 1)
    s_k   : Tensor, (B, N, C, H, W)

    Returns
    -------
    mask_p : Tensor, (B, N, C, H, W)
    mask_n : Tensor, (B, N, C, H, W)
    """
    mask_p = weight_p * spatial_mask  # broadcast on H,W
    mask_n = weight_n * spatial_mask

    return mask_p, mask_n


def compute_mask(mask_p, mask_n):
    """
    Combine positive-/negative masks into a single mask.

    Args
    ----
    mask_p: Tensor, (B, N, C, H, W)
    mask_n: Tensor, (B, N, C, H, W)

    Returns
    -------
    mask : Tensor, (B, N, C, H, W)
           element > 0  → value from mask_p
           element < 0  → value from mask_n
    """
    mask = torch.where(mask_p > 0, mask_p, mask_n)
    return mask
