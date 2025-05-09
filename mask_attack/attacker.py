"""攻击实现

`Attacker` 类是一个用于对 YOLOv8 检测模型进行对抗攻击的工具，实现了 **FGSM（快速梯度符号法）**、**PGD（投影梯度下降法）** 及其 **带掩码版本** 的攻击方法，以生成对抗样本。以下是该类的核心功能描述：

1. **攻击方法**：
   - **FGSM/PGD**：通过计算模型损失相对于输入图像的梯度生成对抗扰动。
   - **带掩码攻击**：利用提供的掩码限制扰动区域（例如仅针对目标边界框内的区域进行攻击）。
   - **批量处理**：支持以批处理方式高效处理数据集，并将结果保存到有序目录中**（没完全实现出来）**。

2. **工作流程**：
   - 使用 `compute_gradient()` 或 `compute_masked_gradient()` 方法计算输入图像的梯度。
   - 通过 `fgsm()`、`masked_fgsm()`、`pgd()` 和 `masked_pgd()` 方法应用扰动生成对抗样本。
   - 将生成的对抗样本和对应标签文件保存到结构化的输出路径（例如，`./mask-attack/method/eps-0.5/` 表示 FGSM 攻击，ε=0.5 的结果）。

3. **主要特性**：
   - **参数化攻击**：支持可调参数，如 `epsilon`（扰动大小）、`alpha`（步长）和 `num_iter`（迭代次数）。
   - **自动化存储**：根据攻击方法、参数和数据集结构组织输出结果。
   - **可扩展性**：预留了对抗训练接口（`adversarial_training()`），方便未来扩展功能。

该类旨在通过对检测模型生成对抗样本来系统评估其鲁棒性，同时保持易用性，支持批量处理和结构化输出管理，便于实验结果的管理和分析。
"""

import os
import shutil
from PIL import Image
from tqdm import tqdm
from ultralytics.utils.loss import v8DetectionLoss
from torch.utils.data import DataLoader
import torch

class Attacker:
    def __init__(self, trainer, dataset, batch_size=1, custom_collate_fn=None):
        """
        Initialize the Attacker
        
        Args:
        """
        self.trainer = trainer
        self.dataset = dataset
        self.custom_collate_fn = custom_collate_fn
        self.batch_size = batch_size
        self.attack_fuction = None
        self.batch_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.custom_collate_fn)
        # ** 

    def compute_gradient(self, sample):
        image = sample['image']
        image.requires_grad = True
        pred = self.trainer.model(image)
        gt_dict = {
            'batch_idx': sample['batch_indices'],
            'cls': sample['classes'],
            'bboxes': sample['bboxes']
        }
        loss_fn = v8DetectionLoss(self.trainer.model)
        loss, _ = loss_fn(pred, gt_dict)
        self.trainer.model.zero_grad()
        loss = loss.sum()
        loss.backward()
        gradient = image.grad.data
        return gradient
        
    def compute_masked_gradient(self, sample):
        image = sample['image']
        mask = sample['mask']
        image.requires_grad = True
        pred = self.trainer.model(image)  # Add batch dimension
        gt_dict = {
            'batch_idx': sample['batch_indices'],
            'cls': sample['classes'],
            'bboxes': sample['bboxes']
        }
        loss_fn = v8DetectionLoss(self.trainer.model)
        loss, _ = loss_fn(pred, gt_dict)
        self.trainer.model.zero_grad()
        loss = loss.sum()
        loss.backward()
        gradient = image.grad.data
        masked_gradient = gradient * mask
        return masked_gradient
    
    def fgsm(self, sample, epsilon):
        image = sample['image']
        gradient = self.compute_gradient(sample)
        if gradient is None:
            return None
        sign_data_grad = gradient.sign()
        perturbed_img = image + epsilon * gradient * sign_data_grad
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
        return perturbed_img

    def masked_fgsm(self, sample, epsilon):
        image = sample['image']
        masked_gradient = self.compute_masked_gradient(sample)
        if masked_gradient is None:
            return None
        # Calculate the sign of the masked gradient
        sign_data_grad = masked_gradient.sign()
        perturbed_img = image + epsilon * masked_gradient * sign_data_grad
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
        return perturbed_img

    def pgd(self, sample, epsilon, alpha, num_iter):
        image = sample['image']
        perturbed_image = image.clone().detach().requires_grad_(True)
        for _ in range(num_iter):
            gradient = self.compute_gradient(sample)
            # Update perturbed image using FGSM-style step
            perturbed_image = perturbed_image + alpha * gradient.sign()
            # Project back to the epsilon-ball and [0, 1] range
            delta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
            perturbed_image = torch.clamp(image + delta, 0, 1).detach().requires_grad_(True)
        return perturbed_image
    
    def masked_pgd(self, sample, epsilon, alpha, num_iter):
        image = sample['image']
        perturbed_image = image.clone().detach().requires_grad_(True)
        for _ in range(num_iter):
            gradient = self.compute_masked_gradient(sample)
            # Update perturbed image using FGSM-style step
            perturbed_image = perturbed_image + alpha * gradient.sign()
            # Project back to the epsilon-ball and [0, 1] range
            delta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
            perturbed_image = torch.clamp(image + delta, 0, 1).detach().requires_grad_(True)
        return perturbed_image
    
    def batch_attack(self, method, output_dir = "./mask-attack/result", **kwargs):
        """
        Perform a batch-wise adversarial attack using the specified method.

        Args:
            epsilon (float): Perturbation magnitude.
            method (str): Attack method to use (e.g., "fgsm_attack", "masked_fgsm_attack").
        """
        # Define available attack methods
        attack_methods = {
            "fgsm": self.fgsm,
            "masked_fgsm": self.masked_fgsm,
            "pgd": self.pgd,
            "masked_pgd": self.masked_pgd
        }
        # Validate the chosen attack method
        if method not in attack_methods:
            raise ValueError(f"Unsupported attack method: {method}. "
                             f"Available methods are: {list(attack_methods.keys())}")
        # Get the selected attack function
        self.attack_function = attack_methods[method]
        print(f"Using attack method: {method}")

        # Create DataLoader for batch processing
        batch_loader_with_progress = tqdm(self.batch_loader, desc="Processing Batches", total=len(self.batch_loader))

        # Process each batch
        for batch in batch_loader_with_progress:
            if batch is None:  # Skip empty batches
                continue
            
            # print(f"batch cls size: {batch['classes'].size()}")
            # print(f"batch bidx size: {batch['batch_indices'].size()}")
            # print(f"batch bbox size: {batch['bboxes'].size()}")
            
            # Perform the attack on the single sample
            if method == "fgsm":
                perturbed_batch = self.attack_function(batch, kwargs["epsilon"])
            elif method == "masked_fgsm":
                perturbed_batch = self.attack_function(batch, kwargs["epsilon"])
            elif method == "pgd":
                perturbed_batch = self.attack_function(batch, kwargs["epsilon"], kwargs["alpha"], kwargs["num_iter"])
            elif method == "masked_pgd":
                perturbed_batch = self.attack_function(batch, kwargs["epsilon"], kwargs["alpha"], kwargs["num_iter"])
            else:
                return
            
            # save pertrubed images
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
            for i in range(len(perturbed_batch)):
                # 单样本处理
                perturbed_img = perturbed_batch[i].cpu()
                original_image_path = batch['image_path'][i]
                original_label_path = batch['label_path'][i]
                # 张量转图像（批量优化：提前转换整个张量再拆分）
                perturbed_np = (perturbed_img * 255).clamp(0, 255).byte().numpy()
                perturbed_np = perturbed_np.transpose(1, 2, 0)  # (H, W, C)
                # 保存图像
                custom_image_path = os.path.join(
                    output_dir, "images", os.path.basename(original_image_path)
                )
                Image.fromarray(perturbed_np, mode='RGB').save(custom_image_path)
                
                # 复制标签文件
                custom_labels_path = os.path.join(
                    output_dir, "labels", os.path.basename(original_label_path)
                )
                shutil.copy2(original_label_path, custom_labels_path)

            # print("Batch-wise masked FGSM attack completed and results saved.")

    
    def adversarial_training(self):
        pass