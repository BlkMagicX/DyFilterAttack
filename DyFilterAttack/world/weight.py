import torch
import numpy as np
from forward import setup_model, extract_features, extract_world_y_det

def compute_gradients(y_det_orig, y_det_target, extract_module_raw_feats):
    """
    Compute gradients of y_det_orig and y_det_target with respect to extract_module_raw_feats.

    Args:
        y_det_orig (Tensor): Original detection tensor with max class probabilities (B, max_out).
        y_det_target (Tensor): Target detection tensor with second max class probabilities (B, max_out).
        extract_module_raw_feats (dict): Raw features extracted from the detect head module.

    Returns:
        grad_orig (dict): Gradients of y_det_orig w.r.t. each feature in extract_module_raw_feats.
        grad_target (dict): Gradients of y_det_target w.r.t. each feature in extract_module_raw_feats.
    """

    for key, feat in extract_module_raw_feats.items():
        if feat is not None and not feat.requires_grad:
            feat.requires_grad_(True)

    grad_orig = {key: None for key in extract_module_raw_feats.keys()}
    grad_target = {key: None for key in extract_module_raw_feats.keys()}

    # y_det_orig梯度计算
    if y_det_orig.requires_grad:
        y_det_orig.sum().backward(retain_graph=True)
        for key, feat in extract_module_raw_feats.items():
            if feat.grad is not None:
                grad_orig[key] = feat.grad.clone()
                feat.grad.zero_()  

    # y_det_target梯度计算
    if y_det_target.requires_grad:
        y_det_target.sum().backward(retain_graph=True)
        for key, feat in extract_module_raw_feats.items():
            if feat.grad is not None:
                grad_target[key] = feat.grad.clone()
                feat.grad.zero_()  

    return grad_orig, grad_target

def compute_channel_weights(grad_orig, grad_target):
    """
    Compute channel weights based on gradients of y_det_orig and y_det_target.

    Args:
        grad_orig (dict): Gradients of y_det_orig w.r.t. each feature in extract_module_raw_feats.
        grad_target (dict): Gradients of y_det_target w.r.t. each feature in extract_module_raw_feats.

    Returns:
        w_k_pos (dict): Positive channel weights (W_k^+) for promoting target class.
        w_k_neg (dict): Negative channel weights (W_k^-) for suppressing original class.
    """
    w_k_pos = {}
    w_k_neg = {}

    for key in grad_orig.keys():
        if grad_orig[key] is None or grad_target[key] is None:
            continue

        # GAP
        alpha_k_co = torch.mean(grad_orig[key], dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        alpha_k_ct = torch.mean(grad_target[key], dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # d_k
        d_k = torch.sign(alpha_k_ct - alpha_k_co)

        # weights
        w_k_neg[key] = d_k * torch.abs(alpha_k_co)
        w_k_pos[key] = d_k * torch.abs(alpha_k_ct)

    return w_k_pos, w_k_neg

def compute_spatial_map(grad_orig):
    """
    Compute spatial map S_k based on gradients of y_det_orig.

    Args:
        grad_orig (dict): Gradients of y_det_orig w.r.t. each feature in extract_module_raw_feats.

    Returns:
        s_k (dict): Spatial maps S_k for each feature in grad_orig.
    """
    s_k = {}

    for key, grad in grad_orig.items():
        if grad is None:
            continue

        grad_abs = torch.abs(grad)  # (B, C, H, W)

        grad_max = torch.max(grad_abs, dim=(2, 3), keepdim=True)[0]  # (B, C, 1, 1)
        
        s_k[key] = grad_abs / (grad_max + 1e-8)  # (B, C, H, W)

    return s_k

def compute_mask(w_k_pos, w_k_neg, s_k):
    """
    Compute masks M_k^+ and M_k^- based on channel weights and spatial maps.

    Args:
        w_k_pos (dict): Positive channel weights (W_k^+) for promoting target class.
        w_k_neg (dict): Negative channel weights (W_k^-) for suppressing original class.
        s_k (dict): Spatial maps S_k for each feature.

    Returns:
        m_k_pos (dict): Positive masks M_k^+ for promoting target class.
        m_k_neg (dict): Negative masks M_k^- for suppressing original class.
    """
    m_k_pos = {}
    m_k_neg = {}

    for key in s_k.keys():
        if key not in w_k_pos or key not in w_k_neg:
            continue

        m_k_pos[key] = w_k_pos[key] * s_k[key]  # (B, C, H, W)
        m_k_neg[key] = w_k_neg[key] * s_k[key]  # (B, C, H, W)

    return m_k_pos, m_k_neg

if __name__ == "__main__":
    
    model_path = './DyFilterAttack/models/yolov8s-world.pt'
    image_path = './DyFilterAttack/analyzer/bus.jpg'

    yolo_world, world_layers = setup_model(model_path=model_path, verbose=True)
    extract_module_raw_feats, extract_module = extract_features(
        yolo=yolo_world,
        layers=world_layers,
        image_path=image_path,
        extract_module_name='detect_head'
    )

    y_det_orig, y_det_target = extract_world_y_det(
        world=yolo_world,
        layers=world_layers,
        detect_head_raw_feats=extract_module_raw_feats
    )

    grad_orig, grad_target = compute_gradients(y_det_orig, y_det_target, extract_module_raw_feats)

    w_k_pos, w_k_neg = compute_channel_weights(grad_orig, grad_target)

    s_k = compute_spatial_map(grad_orig)

    m_k_pos, m_k_neg = compute_mask(w_k_pos, w_k_neg, s_k)