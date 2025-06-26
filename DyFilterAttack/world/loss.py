import torch


def total(mask_p, mask_n, activation, lambda_promote=1.0, lambda_surpress=1.0):
    """
    Compute the total loss for adversarial attack, combining L_promote and L_suppress.

    Args:
        mask_p (dict): Positive masks M_k^+ for promoting target class.
        mask_n (dict): Negative masks M_k^- for suppressing original class.
        activation (dict): Raw features extracted from the detect head module.
        lambda_weight (float): Weight for balancing L_suppress in total loss. Default is 1.0.

    Returns:
        loss_total (Tensor): Total loss L_total = L_promote + lambda * L_suppress.
        loss_promote (Tensor): Loss for promoting target class.
        loss_suppress (Tensor): Loss for suppressing original class.
    """
    loss_promote = 0.0
    loss_suppress = 0.0

    for key, feat in activation.items():
        if feat is not None and not feat.requires_grad:
            feat.requires_grad_(True)

    for key in mask_p.keys():
        if key not in activation or key not in mask_n:
            continue

        feat = activation[key]  # (B, C, H, W)
        m_pos = mask_p[key]  # (B, C, H, W)
        m_neg = mask_n[key]  # (B, C, H, W)

        loss_promote += -torch.sum(m_pos * feat)

        loss_suppress += -torch.sum(m_neg * feat)

    loss_total = lambda_promote * loss_promote + lambda_surpress * loss_suppress

    return loss_total, loss_promote, loss_suppress
