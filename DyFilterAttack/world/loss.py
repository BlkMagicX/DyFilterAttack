import torch
from forward import setup_model, extract_features, extract_world_y_det
from weight import compute_gradients, compute_channel_weights, compute_spatial_map, compute_mask

def compute_loss(m_k_pos, m_k_neg, extract_module_raw_feats, lambda_weight=1.0):
    """
    Compute the total loss for adversarial attack, combining L_promote and L_suppress.

    Args:
        m_k_pos (dict): Positive masks M_k^+ for promoting target class.
        m_k_neg (dict): Negative masks M_k^- for suppressing original class.
        extract_module_raw_feats (dict): Raw features extracted from the detect head module.
        lambda_weight (float): Weight for balancing L_suppress in total loss. Default is 1.0.

    Returns:
        loss_total (Tensor): Total loss L_total = L_promote + lambda * L_suppress.
        loss_promote (Tensor): Loss for promoting target class.
        loss_suppress (Tensor): Loss for suppressing original class.
    """
    loss_promote = 0.0
    loss_suppress = 0.0

    for key, feat in extract_module_raw_feats.items():
        if feat is not None and not feat.requires_grad:
            feat.requires_grad_(True)

    for key in m_k_pos.keys():
        if key not in extract_module_raw_feats or key not in m_k_neg:
            continue

        feat = extract_module_raw_feats[key]  # (B, C, H, W)
        m_pos = m_k_pos[key]  # (B, C, H, W)
        m_neg = m_k_neg[key]  # (B, C, H, W)

        loss_promote += -torch.sum(m_pos * feat)

        loss_suppress += -torch.sum(m_neg * feat)

    loss_total = loss_promote + lambda_weight * loss_suppress

    return loss_total, loss_promote, loss_suppress

if __name__ == "__main__":
    model_path = './DyFilterAttack/models/yolov8s-world.pt'
    image_path = './DyFilterAttack/analyzer/bus.jpg'
    lambda_weight = 1.0  

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

    loss_total, loss_promote, loss_suppress = compute_loss(
        m_k_pos, m_k_neg, extract_module_raw_feats, lambda_weight
    )
    
    print("\nLoss values:")
    print(f"L_promote: {loss_promote.item():.4f}")
    print(f"L_suppress: {loss_suppress.item():.4f}")
    print(f"L_total: {loss_total.item():.4f}")
