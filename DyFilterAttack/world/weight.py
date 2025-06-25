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
    alpha_o : Tensor, shape (B, N, C, 1, 1)
            promote target class
    alpha_t : Tensor, shape (B, N, C, 1, 1)
            suppress original class
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

    weight_p = direction * alpha_t.abs()
    weight_n = direction * alpha_o.abs()

    return weight_p, weight_n


def compute_mask(weight_p: torch.Tensor, weight_n: torch.Tensor, spatial_mask: torch.Tensor):
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


if __name__ == "__main__":

    model_path = "./DyFilterAttack/models/yolov8s-world.pt"
    image_path = "./DyFilterAttack/testset/bus.jpg"

    # Initialize model and image
    yolo_world, world_layers = setup_model(model_path=model_path, verbose=True)
    image_tensor = preprocess_image(image_path, yolo_world.device)
    grad_orig_A, grad_target_A, A_value = compute_gradients_y_det_and_activation(
        world=yolo_world, image_tensor=image_tensor, target_layer_name='model.model.22.cv2'
    )

    weight_p, weight_n = compute_channel_weights(grad_orig_A, grad_target_A)
    s_k = compute_spatial_map(grad_orig_A)
    mask_p, mask_n = compute_mask(weight_p, weight_n, s_k)

    print("weight_p  :", weight_p.shape)  # (B,N,C,1,1)
    print("s_k    :", s_k.shape)  # (B,N,C,H,W)
    print("mask_p  :", mask_p.shape)  # (B,N,C,H,W)
