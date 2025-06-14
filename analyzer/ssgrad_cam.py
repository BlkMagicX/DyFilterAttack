from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import ttach as tta

from pytorch_grad_cam.base_cam import BaseCAM 
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class SSGradCAM(BaseCAM):
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
        tta_transforms: Optional[tta.Compose] = None,
        detach: bool = True
    ) -> None:
        super(SSGradCAM, self).__init__(
            model, target_layers, reshape_transform, compute_input_gradient, uses_gradients, tta_transforms, detach
        )
    
    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[Callable],
        activations: torch.Tensor,
        grads: torch.Tensor
    ) -> np.ndarray:
        """
        重写BaseCAM的get_cam_weights函数: 
        包括下列操作: 
        1. 计算alpha, 执行GAP以获得表示该通道捕获特征重要性的权重
        2. 计算空间图S, 修改alpha
        """
        if isinstance(grads, torch.Tensor):
            grads = grads.cpu().detach().numpy()
        # GAP：对空间维度（高度和宽度）取平均值
        if len(grads.shape) == 4:
            alpha = np.mean(grads, axis=(2, 3))
        else:
            raise ValueError("Invalid grads shape.")
        grads_abs = np.abs(grads)
        max_grads = np.max(grads_abs, axis=(2, 3), keepdims=True)
        space_map = grads_abs / (max_grads + 1e-8)
        weights = alpha[:, :, None, None] * space_map
        return weights
    
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[Callable],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False
    ) -> np.ndarray:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        # Convert activations to numpy if needed
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().detach().numpy()  # Shape: [batch_size, channels, height, width]
        # Convert grads to numpy if needed
        if isinstance(grads, torch.Tensor):
            grads = grads.cpu().detach().numpy()
        cam = weights * activations
        # softmax
        cam = np.sum(cam, axis=1)
        cam = np.maximum(cam, 0)
        return cam

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[Callable],
        eigen_smooth: bool
    ) -> List[np.ndarray]:
        """
        Compute and normalize heatmaps for each target layer, 
        and calculate channel sensitivity scores.
        """
        if self.detach:
            activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
            grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        else:
            activations_list = [a for a in self.activations_and_grads.activations]
            grads_list = [g for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []
        self.channel_sensitivity = []  # Store sensitivity scores
        for i, target_layer in enumerate(self.target_layers):
            layer_activations = activations_list[i] if i < len(activations_list) else None
            layer_grads = grads_list[i] if i < len(grads_list) else None
            if layer_activations is None or layer_grads is None:
                continue
            weights = self.get_cam_weights(input_tensor, target_layer, targets, layer_activations, layer_grads)
            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            
            # Calculate channel sensitivity scores
            channel_scores = np.mean(weights, axis=(2, 3))  # Global Average Pooling
            normalized_scores = channel_scores / (np.max(channel_scores, axis=1, keepdims=True) + 1e-8)  # Max normalization
            
            # Store scores and process CAM
            self.channel_sensitivity.append(normalized_scores.squeeze(0))  # Remove batch dimension
            
            # Normalize and scale CAM
            max_val = np.max(cam, axis=(1, 2), keepdims=True)
            cam = cam / (max_val + 1e-8)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
        
        print(self.channel_sensitivity)
        return cam_per_target_layer

    # Modification 4: Override aggregate_multi_layers to resize and sum heatmaps as per Eq. 6
    def aggregate_multi_layers(self, cam_per_target_layer: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate heatmaps from multiple layers by resizing and summing them.
        """
        if not cam_per_target_layer:
            raise ValueError("No CAMs to aggregate.")
        
        # Stack and sum the normalized, resized heatmaps
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)  # Shape: [batch_size, num_layers, H, W]
        result = np.sum(cam_per_target_layer, axis=1)  # Sum over layers
        return scale_cam_image(result)

    # Note: The forward method remains largely unchanged but relies on the overridden methods above.
    def forward(
        self,
        input_tensor: torch.Tensor,
        targets: List[Callable],
        eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(self.outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum(target(output) for target, output in zip(targets, self.outputs))
            if self.detach:
                loss.backward(retain_graph=True)
            else:
                torch.autograd.grad(loss, input_tensor, retain_graph=True, create_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)
    
if __name__ == "__main__":
    import torch.nn as nn
    from torchvision.models import resnet18
    
    model = resnet18(pretrained=True).eval()
    target_layers = [model.layer4[-1]]  # Example layer
    cam = SSGradCAM(model=model, target_layers=target_layers)
    input_tensor = torch.rand(1, 3, 224, 224)
    targets = [ClassifierOutputTarget(281)]  # Example target class
    
    # 获取CAM和敏感度得分
    grayscale_cam = cam(input_tensor, targets)
    sensitivity_scores = cam.channel_sensitivity  # 每个目标层的敏感度得分
    
    print("Channel Sensitivity Scores (0-1 normalized):")
    print(sensitivity_scores[0])  # 第一个目标层的通道敏感度
    print(f"Scores range: [{sensitivity_scores[0].min():.4f}, {sensitivity_scores[0].max():.4f}]")
