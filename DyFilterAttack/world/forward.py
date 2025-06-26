import torch
from ultralytics.utils import ops


def compute_gradients_y_det_and_activation(world, image_tensor, target_layer_name, det_ids_batch, predictor):
    """
    Compute gradients under the “fixed-target” policy.

    Args:
        world (YOLOWorld): Model.
        image_tensor (Tensor): Current adversarial sample.
        target_layer_name (str): Name of the layer to attack.

    Returns:
        Tuple: (grad_A_orig, grad_A_target, activations['value'])
    """

    # ! Data Flow Explanation:
    # 1. Start with YOLOWorld inference to get predictions (preds)
    #    └── preds contain both bounding box and class logits
    # 2. Apply Non-Maximum Suppression (NMS) to filter overlapping detections
    #    └── Get keep_idxs: indices of the kept detection boxes after NMS
    # 3. Create an initial valid_mask, set all to True
    #    └── This mask indicates which detection boxes are valid
    # 4. Further filter valid_mask using det_ids_batch
    #    └── Only keep the boxes that we are interested in attacking or analyzing
    # 5. Based on the filtered indices, extract y_det (class logits of selected boxes)
    #    └── For each selected box:
    #        - Take Top1 class (most confident) → used for grad_orig
    #        - Take Top2 class (second most confident) → used for grad_target
    # 6. Perform two separate backward passes:
    #    ┐
    #    ├─ Backward on Top1 class scores → compute grad_orig
    #    └─ Backward on Top2 class scores → compute grad_target

    device = image_tensor.device
    world.zero_grad()

    # Capture the activation of the target layer and keep its grad
    activations = {}

    def forward_hook(module, input, output):
        output.retain_grad()
        activations["value"] = output

    target_module = world
    for part in target_layer_name.split("."):
        target_module = getattr(target_module, part)
    handle = target_module.register_forward_hook(forward_hook)

    # Forward pass (with graph) and NMS

    preds, _ = world.model(image_tensor)  # (B, 4+nc, N)
    raw_cls = preds[:, 4:, :]  # (B, nc, N)

    detections, keep_idxs = ops.non_max_suppression(
        preds,
        predictor.args.conf,
        predictor.args.iou,
        predictor.args.classes,
        predictor.args.agnostic_nms,
        predictor.args.max_det,
        nc=0 if predictor.args.task == "detect" else len(predictor.model.names),
        end2end=getattr(predictor.model, "end2end", False),
        rotated=predictor.args.task == "obb",
        return_idxs=True,
    )

    # Pad variable-length indices to max_out
    max_out = max(idx.numel() for idx in keep_idxs)
    nms_idxs = raw_cls.new_zeros(raw_cls.shape[0], max_out, dtype=torch.long)
    valid_mask = torch.zeros(raw_cls.shape[0], max_out, dtype=torch.bool, device=raw_cls.device)

    for b, ids in enumerate(keep_idxs):
        if ids.numel():
            nms_idxs[b, : ids.numel()] = ids
            valid_mask[b, : ids.numel()] = True

    if det_ids_batch is not None:
        for b, sel in enumerate(det_ids_batch):
            if sel.numel():
                mask = torch.isin(nms_idxs[b, :], sel.to(device))
                valid_mask[b] &= mask

    if not valid_mask.any():
        handle.remove()
        raise RuntimeError("No valid detections selected for gradient computation.")

    gather_idx = nms_idxs.unsqueeze(1).expand(-1, raw_cls.shape[1], -1)  # (B, nc, max_out)
    y_det = torch.gather(raw_cls, 2, gather_idx).masked_fill(~valid_mask.unsqueeze(1), 0.0)

    if y_det.shape[2] == 0:  # no detections
        print("Warning: no targets to attack.")
        handle.remove()
        return None, None, activations.get("value")

    # Pick top-1 / top-2 class scores per detection
    _, topk_idx = torch.topk(y_det, 2, dim=1)
    y_det_orig = y_det.gather(1, topk_idx[:, :1, :]).squeeze(1)  # (B, max_out)
    y_det_target = y_det.gather(1, topk_idx[:, 1:, :]).squeeze(1)  # (B, max_out)

    # --- per-scalar gradients ---
    A = activations["value"]  # (B, C, H, W)
    Bsz, Cch, H, W = A.shape
    grad_orig = A.new_zeros(Bsz, max_out, Cch, H, W)
    grad_target = A.new_zeros(Bsz, max_out, Cch, H, W)

    for b in range(Bsz):
        for n in range(max_out):
            if not valid_mask[b, n]:
                continue

            # grad of top-1 score
            world.zero_grad(set_to_none=True)
            torch.autograd.grad(y_det_orig[b, n], A, retain_graph=True)
            grad_orig[b, n] = A.grad[b].detach()
            A.grad.zero_()

            # grad of top-2 score
            torch.autograd.grad(y_det_target[b, n], A, retain_graph=True)
            grad_target[b, n] = A.grad[b].detach()
            A.grad.zero_()

    handle.remove()
    print("Gradients computed.")

    return grad_orig, grad_target, A.detach()
