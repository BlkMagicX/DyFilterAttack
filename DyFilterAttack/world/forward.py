import numpy as np
import torch
import warnings
from pathlib import Path
from ultralytics import YOLOWorld
from ultralytics.utils import ops
from DyFilterAttack.analyzer.utils import SaveFeatures
from PIL import Image
import torchvision.transforms as T
import numpy as np

# sys.modules.clear()


def preprocess_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device)
    image_tensor.requires_grad = True
    return image_tensor


def setup_model(model_path, verbose=True):
    """
    Initialize YOLO model and image for inference.

    Args:
        model_path (str): Path to the trained YOLO model.
        verbose (bool): Whether to print additional information during initialization.

    Returns:
        yolo (YOLOWorld): Initialized YOLO model.
        layers (dict): Dictionary containing the layers used in the YOLO model for feature extraction.
    """
    print("\nsetup_model...")
    warnings.filterwarnings(action="ignore")
    warnings.simplefilter(action="ignore")

    path = Path(model_path if isinstance(model_path, (str, Path)) else "")
    if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:
        world = YOLOWorld(path, verbose=verbose)
        world = world.to(torch.device(device="cuda" if torch.cuda.is_available() else "cpu"))

        key_layer_idx = {
            "backbone_c2f1": 2,
            "backbone_c2f2": 4,
            "backbone_c2f3": 6,
            "backbone_c2f4": 8,
            "backbone_sppf": 9,
            "neck_c2f1": 15,
            "neck_c2f2": 19,
            "neck_c2f3": 22,
            "detect_head": 23,
        }

        layers = {layer: world.model.model[idx] for layer, idx in key_layer_idx.items()}

    return world, layers


# ! Deprecated
def choose_extract_static_module(layers, extract_module_name):
    # Hook for extracting features
    save_feats = SaveFeatures()
    extract_module = layers[extract_module_name]
    save_feats.register_hooks(extract_module, parent_path=extract_module_name, verbose=True)
    return save_feats, extract_module


# ! Deprecated
def extract_static_features(yolo, image_path, save_feats):
    """
    Extract features from the image using the YOLO model.

    Args:
        yolo (YOLOWorld): Initialized YOLO model.
        layers (dict): Dictionary containing layers for feature extraction.
        image_path (str): Path to the input image.
        extract_module_name (str): Name of the module from which to extract features.

    Returns:
        extract_module_raw_feats (dict): Extracted features from the detection head.
        extract_module (torch.nn.Module): The specified module from which the features are extracted.
    """

    print("\nextract_features...")
    results = yolo.predict(image_path)
    result = results[0]

    for det in result.boxes:
        xmin, ymin, xmax, ymax = det.xyxy[0]
        conf = det.conf  # Confidence
        cls = det.cls  # Class ID
        class_name = result.names[cls[0].item()]
        print(f"bbox: {xmin}, {ymin}, {xmax}, {ymax}, conf: {conf}, class: {class_name}")

    extract_module_raw_feats = save_feats.get_features()

    return extract_module_raw_feats


# ! Deprecated
def extract_static_world_y_det(world, layers, save_feats):
    """
    Extract the y_det and y_det_target from the model outputs.

    Args:
        layers (dict): Dictionary containing the layers for feature extraction.
        save_feats.features (dict): Raw features extracted from the detect head module.

    Returns:
        y_det_orig (Tensor): The original detection tensor with max class probabilities.
        y_det_target (Tensor): The detection tensor with second max class probabilities.
    """
    # Retrieve the actual module object using the module name
    detect_head = layers["detect_head"]

    # Extract necessary feature values for further processing
    nl = detect_head.nl
    nc, reg_max = detect_head.nc, detect_head.reg_max
    no = nc + 4 * reg_max
    assert no == detect_head.no

    # process1
    # text -> (B, nc, embed_dim)
    # image -> (B, embed_dim, H, W)
    # cv4 contrast(iamge, text) -> (B, nc, H, W)
    # cv2(image) -> (B, reg_max * 4, H, W)
    # cat_result -> (B, nc + reg_max * 4, H ,W) -> (B, no, H ,W)
    # x[i] -> cat_result[i] (i = 1, 2, nl)
    features = save_feats.features
    cv2_raw_feats = [features[f"detect_head.cv2.{i}"] for i in range(nl)]
    cv4_raw_feats = [features[f"detect_head.cv4.{i}"] for i in range(nl)]

    # process2 (_inference)
    # flat(x[i]) -> (B, no, H * W)
    # cat(x) -> (B, C, H0 * W0 + H1 * W1 + H2 * W2)
    # split(x) -> bbox(B, 4 * reg_max, H0 * W0 + H1 * W1 + H2 * W2), cls(logit)(B, nc, H0 * W0 + H1 * W1 + H2 * W2)
    # docode(bbox) -> dbox
    # logit(cls) -> sigmoid(cls)
    # y -> cat(dbox, cls)
    cat_raw_feats = [torch.cat((cv2_raw_feats[i], cv4_raw_feats[i]), 1) for i in range(nl)]
    flatten_raw_feats = torch.cat([cat_raw_feat.view(cat_raw_feats[0].shape[0], nc + 4 * reg_max, -1) for cat_raw_feat in cat_raw_feats], 2)
    raw_box = flatten_raw_feats[:, : reg_max * 4]
    raw_cls = flatten_raw_feats[:, reg_max * 4 :]

    # Decode bounding boxes and logits (classification)
    dfl_feats = features["detect_head.dfl"]
    dbox = detect_head.decode_bboxes(dfl_feats, detect_head.anchors.unsqueeze(0)) * detect_head.strides
    logit_cls = raw_cls
    sigmoid_cls = logit_cls.sigmoid()

    # process3 (construct y_det_orig and y_de_target)
    # obtain the specific cls indices selected by non_max_suppression
    preds = torch.cat([dbox, sigmoid_cls], 1)  # (B, 4+nc, N)
    predictor = world.predictor
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

    # Get the number of NMS outputs
    num_nms_output = [idx.numel() for idx in keep_idxs]
    max_out = max(num_nms_output)

    # Initialize a tensor for y_det
    y_det = raw_cls.new_zeros(raw_cls.shape[0], raw_cls.shape[1], max_out)
    for b, idx in enumerate(keep_idxs):
        if idx.numel() > 0:
            y_det[b, :, : idx.numel()] = flatten_raw_feats[: raw_cls.shape[0], raw_box.shape[1] :, idx]

    # Get the first max class index (for y_det_orig)
    first_max_cls_idx = torch.argmax(y_det, dim=1)  # (B, max_out)
    y_det_orig = y_det[torch.arange(y_det.shape[0]), first_max_cls_idx, torch.arange(y_det.shape[2])]  # (B, max_out)

    # Get the second max class index (for y_det_target)
    _, topk_indices = torch.topk(y_det, 2, dim=1)
    second_max_cls_idx = topk_indices[:, 1]  # (B, max_out)
    y_det_target = y_det[torch.arange(y_det.shape[0]), second_max_cls_idx, torch.arange(y_det.shape[2])]  # (B, max_out)

    print("\nextract_y_det...")
    print("shapes:")
    print(f"y_det_orig       {y_det_orig.size()}")
    print(f"y_det_target     {y_det_target.size()}")
    # print(f"gradients        {gradients.size()}")
    print("indexs:")
    print(f"y_det_orig       {first_max_cls_idx}")
    print(f"y_det_target     {second_max_cls_idx}")
    print("logits:")
    print(f"y_det_orig       {y_det_orig}")
    print(f"y_det_target     {y_det_target}")
    print("sigmoid:")
    print(f"y_det_orig       {y_det_orig.sigmoid()}")
    print(f"y_det_target     {y_det_target.sigmoid()}")

    return y_det_orig, y_det_target


def compute_gradients_y_det_and_activation(world, image_tensor, target_layer_name):
    """
    在“固定目标”策略下计算梯度。

    Args:
        world (YOLOWorld): 模型。
        image_tensor (Tensor): 当前的对抗样本张量。
        target_indices (Tensor): 从干净图像上得到的、固定的目标索引。
        target_layer_name (str): 目标层的名称。

    Returns:
        Tuple: (grad_A_orig, grad_A_target, activations['value'])
    """

    world.zero_grad()

    def forward_hook(module, input, output):
        output.retain_grad()
        activations['value'] = output

    activations = {}
    module_path = target_layer_name.split('.')
    target_module = world
    for part in module_path:
        target_module = getattr(target_module, part)
    handle = target_module.register_forward_hook(forward_hook)

    # N: num of anchors
    world.predictor = world._smart_load("predictor")(_callbacks=world.callbacks)
    world.predictor.setup_model(model=world.model, verbose=False)
    predictor = world.predictor

    preds, _ = world.model(image_tensor)  # pred (B, 4+nc, N)
    raw_cls = preds[:, 4:, :]  #  logits (B, nc, N)
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

    max_out = max(idx.numel() for idx in keep_idxs)

    nms_idxs = raw_cls.new_zeros(raw_cls.shape[0], max_out, dtype=torch.long)  # (B, max_out)
    mask = torch.zeros(raw_cls.shape[0], max_out, dtype=torch.bool, device=raw_cls.device)

    for b, ids in enumerate(keep_idxs):
        if ids.numel() == 0:
            continue
        nms_idxs[b, : ids.numel()] = ids
        mask[b, : ids.numel()] = True

    nms_idxs = nms_idxs.unsqueeze(1).expand(-1, raw_cls.shape[1], -1)  # (B, nc, max_out)
    y_det = torch.gather(raw_cls, 2, nms_idxs)  # (B, nc, max_out)
    y_det = y_det.masked_fill(~mask.unsqueeze(1), 0.0)
    nms_idxs = nms_idxs[:, 0, :]

    if y_det.shape[2] == 0:  # 如果没有检测到任何目标
        print("Warning: No targets were provided to attack.")
        handle.remove()
        return None, None, activations.get('value')

    _, topk_indices = torch.topk(y_det, 2, dim=1)

    y_det_orig = y_det.gather(1, topk_indices[:, :1, :]).squeeze(1)  # Shape: [1, num_fixed_targets]
    y_det_target = y_det.gather(1, topk_indices[:, 1:, :]).squeeze(1)  # Shape: [1, num_fixed_targets]

    total_loss_orig = y_det_orig.sum()
    total_loss_target = y_det_target.sum()

    total_loss_orig = y_det_orig.sum()
    total_loss_target = y_det_target.sum()

    print(f"\nScores for fixed targets (sum): orig={total_loss_orig.item()}, target={total_loss_target.item()}")

    # backforward
    total_loss_orig.backward(retain_graph=True)
    grad_orig_A = activations['value'].grad.clone()

    world.zero_grad()
    total_loss_target.backward()
    grad_target_A = activations['value'].grad.clone()

    A_value = activations.get('value')
    handle.remove()
    print("Successfully computed gradients on fixed targets.")

    return grad_orig_A, grad_target_A, A_value


if __name__ == "__main__":

    model_path = "./DyFilterAttack/models/yolov8s-world.pt"
    image_path = "./DyFilterAttack/testset/bus.jpg"

    # Initialize model and image
    yolo_world, world_layers = setup_model(model_path=model_path, verbose=True)
    image_tensor = preprocess_image(image_path, yolo_world.device)

    # save_feats, extract_module = choose_extract_static_module(layers=world_layers, extract_module_name='detect_head')
    # activations = extract_static_features(yolo=yolo_world, image_path=image_path, save_feats=save_feats)
    # y_det_orig, y_det_target = extract_static_world_y_det(world=yolo_world, layers=world_layers, save_feats=save_feats)
    # compute_gradients_y_det_and_activation(layer_name='detect_head.cv2.0', y_det_orig=y_det_orig, y_det_target=y_det_target, save_feats=save_feats)

    grad_orig_A, grad_target_A, A_value = compute_gradients_y_det_and_activation(
        world=yolo_world, image_tensor=image_tensor, target_layer_name='model.model.22.cv2'
    )

    print(grad_orig_A.size())
    print(grad_target_A.size())
    print(A_value.size())
