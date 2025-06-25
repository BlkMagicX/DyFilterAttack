import torch
import warnings
from pathlib import Path
from ultralytics import YOLOWorld


import forward
import loss
import weight


class WorldAttacker:
    def __init__(self, model_path, attack_layer_name):
        self.world, self.layers = self.setup_model(model_path, verbose=True)
        self.choose_attack_layer(attack_layer_name)

    def setup_model(self, model_path, verbose=True):
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

    def choose_attack_layer(self, attack_layer_name):
        self.attack_layer_name = attack_layer_name
        self.attack_layer = self.layers["attack_layer_name"]

    def forward(self, image_path, target_layer_name='model.model.22.cv2'):
        # ! 这里暂时是 image_path, 后面要修改成image_adv的tensor向量
        self.image_tensor = forward.preprocess_image(image_path, self.world.device)
        # compute activation and gard
        self.grad_orig, self.grad_target, self.activation = forward.compute_gradients_y_det_and_activation(
            world=self.world, image_tensor=self.image_tensor, target_layer_name=target_layer_name
        )
        # compute parameters
        self.alpha_o, self.alpha_t = weight.compute_basic_weights(grad_orig=self.grad_orig, grad_target=self.grad_target)
        self.spatial_mask = weight.compute_spatial_mask(grad_orig=self.grad_orig)
        self.direction = weight.compute_direction(alpha_o=self.alpha_o, alpha_t=self.alpha_t)
        self.weight_p, self.weight_n = weight.compute_full_weight(alpha_o=self.alpha_o, alpha_t=self.alpha_t, direction=self.direction)
        self.mask_p, self.mask_n = weight.compute_decouple_mask(weight_p=self.weight_p, weight_n=self.weight_n, spatial_mask=self.spatial_mask)
        self.mask = weight.compute_mask(mask_p=self.mask_p, mask_n=self.mask_n)
        loss_total, loss_promote, loss_surpress = loss.total(self.mask_p, self.mask_n, self.activation)

    def backward(self):
        pass

    def adversarial_attack(self):
        pass

    def batch_attack(self):
        pass

    def evaluation(self):
        pass
