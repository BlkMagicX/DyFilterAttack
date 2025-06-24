import os
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLOWorld
from DyFilterAttack.analyzer.utils import SaveFeatures
from ultralytics.utils import ops
import warnings
import matplotlib.pyplot as plt


def setup_model(model_path, verbose):
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
        yolo = YOLOWorld(path, verbose=verbose)
        yolo = yolo.to(torch.device(device="cuda" if torch.cuda.is_available() else "cpu"))
        
        key_layer_idx = {
            "backbone_c2f1": 2,
            "backbone_c2f2": 4,
            "backbone_c2f3": 6,
            "backbone_c2f4": 8, 
            "backbone_sppf": 9,
            "neck_c2f1": 15,
            "neck_c2f2": 19,
            "neck_c2f3": 22,
            "detect_head": 23
        }
        
        layers = {layer: yolo.model.model[idx] for layer, idx in key_layer_idx.items()}
        
    return yolo, layers


def extract_features(yolo, layers, image_path, extract_module_name: str):
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
    # Hook for extracting features
    save_feats = SaveFeatures()
    extract_module = layers[extract_module_name]  # Directly get module from layers using name
    save_feats.register_hooks(module=extract_module, parent_path=extract_module_name, verbose=True)
    
    # Perform inference on the input image
    # plot det result[B=0]
    print("\nextract_features...")
    results = yolo.predict(image_path)
    result = results[0]
    for det in result.boxes:
        xmin, ymin, xmax, ymax = det.xyxy[0]
        conf = det.conf  # Confidence
        cls = det.cls  # Class ID
        class_name = result.names[cls[0].item()]
        print(f"bbox: {xmin}, {ymin}, {xmax}, {ymax}, conf: {conf}, class: {class_name}")

    image = Image.fromarray(result.plot()[:, :, ::-1])
    image.show()
    image.save('./DyFilterAttack/world/result/bus_result.jpg')
    
    extract_module_raw_feats = save_feats.get_features()

    return extract_module_raw_feats, extract_module


def extract_y_det(layers, extract_module_name, extract_module_raw_feats):
    """
    Extract the y_det and y_det_target from the model outputs.
    
    Args:
        layers (dict): Dictionary containing the layers for feature extraction.
        extract_module_name (str): Name of the module from which to extract features.
        extract_module_raw_feats (dict): Raw features extracted from the specified module.

    Returns:
        y_det_orig (Tensor): The original detection tensor with max class probabilities.
        y_det_target (Tensor): The detection tensor with second max class probabilities.
    """
    # Retrieve the actual module object using the module name
    extract_module = layers[extract_module_name]
    
    # Extract necessary feature values for further processing
    nl = extract_module.nl
    nc, reg_max =  extract_module.nc, extract_module.reg_max
    no = nc + 4 * reg_max
    assert no == extract_module.no
    
    # Process 1: Extract raw features for CV2 and CV4
    cv2_raw_feats = [extract_module_raw_feats[f'{extract_module_name}.cv2.{i}'] for i in range(nl)]
    cv4_raw_feats = [extract_module_raw_feats[f'{extract_module_name}.cv4.{i}'] for i in range(nl)]

    # Process 2: Flatten and concatenate features
    cat_raw_feats = [torch.cat((cv2_raw_feats[i], cv4_raw_feats[i]), 1) for i in range(nl)]
    flatten_raw_feats = torch.cat([cat_raw_feat.view(cat_raw_feats[0].shape[0], nc + 4 * reg_max, -1) for cat_raw_feat in cat_raw_feats], 2)
    raw_box = flatten_raw_feats[:, : reg_max * 4]
    raw_cls = flatten_raw_feats[:, reg_max * 4 :]

    # Decode bounding boxes and logits (classification)
    dfl_feats = extract_module_raw_feats[f'{extract_module_name}.dfl']
    dbox = extract_module.decode_bboxes(dfl_feats, extract_module.anchors.unsqueeze(0)) * extract_module.strides
    logit_cls = raw_cls
    sigmoid_cls = logit_cls.sigmoid()

    # Apply non-max suppression
    preds = torch.cat([dbox, sigmoid_cls], 1)  # (B, 4+nc, N)
    predictor = yolo.predictor
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
            y_det[b, :, :idx.numel()] = flatten_raw_feats[:raw_cls.shape[0], raw_box.shape[1]:, idx]

    # Get the first max class index (for y_det_orig)
    first_max_cls_idx = torch.argmax(y_det, dim=1)  # (B, max_out)
    y_det_orig = y_det[torch.arange(y_det.shape[0]), first_max_cls_idx, torch.arange(y_det.shape[2])]  # (B, max_out)

    # Get the second max class index (for y_det_target)
    _, topk_indices = torch.topk(y_det, 2, dim=1)
    second_max_cls_idx = topk_indices[:, 1]  # (B, max_out)
    y_det_target = y_det[torch.arange(y_det.shape[0]), second_max_cls_idx, torch.arange(y_det.shape[2])]  # (B, max_out)
    
    print("\nextract_y_det...")
    print("shapes:")
    print(f'y_det_orig       {y_det_orig.size()}')
    print(f'y_det_target     {y_det_target.size()}')
    print("indexs:")
    print(f'y_det_orig       {first_max_cls_idx}')
    print(f'y_det_target     {second_max_cls_idx}')
    print("logits:")
    print(f'y_det_orig       {y_det_orig}')
    print(f'y_det_target     {y_det_target}')
    print("sigmoid:")
    print(f'y_det_orig       {y_det_orig.sigmoid()}')
    print(f'y_det_target     {y_det_target.sigmoid()}')

    return y_det_orig, y_det_target


if __name__ == "__main__":
    
    model_path = './DyFilterAttack/models/yolov8s-world.pt'
    image_path = './DyFilterAttack/testset/bus.jpg'
    
    # Initialize model and image
    yolo, yolo_layers = setup_model(model_path=model_path, verbose=True)
    
    extract_module_raw_feats, extract_module = extract_features(yolo=yolo, 
                                                                layers=yolo_layers, 
                                                                image_path=image_path, 
                                                                extract_module_name='detect_head')
    
    y_det_orig, y_det_target = extract_y_det(layers=yolo_layers,
                                             extract_module_name='detect_head',
                                             extract_module_raw_feats=extract_module_raw_feats)
