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
from mask_attack.utils.dataset import custom_collate_fn
import torch

class Attacker:
    def __init__(self, trainer, dataset, batch_size=1, custom_collate_fn=custom_collate_fn):
        """
        Initialize the Attacker
        
        Args:
        """
        self.trainer = trainer
        self.dataset = dataset
        self.custom_collate_fn = custom_collate_fn
        self.batch_size = batch_size
        self.batch_loader = DataLoader(self.dataset, batch_size=self.batch_size, 
                                       shuffle=False, collate_fn=self.custom_collate_fn)
        # param view
        self.method_config = {
            "fgsm": {
                'attack_function': self.fgsm,
                'params_name': ["epsilon"]
            },
            "masked_fgsm": {
                'attack_function': self.masked_fgsm,
                'params_name': ["epsilon"]
            },
            "pgd": {
                'attack_function': self.pgd,
                'params_name': ["epsilon", "alpha", "num_iter"]
            },
            "masked_pgd":{
                'attack_function': self.masked_pgd,
                'params_name': ["epsilon", "alpha", "num_iter"]
            },
            "filter_attack":{
                'attack_function': self.filter_attack,
                'params_name': ["lr", "epsilon", "num_iter", "key_filters"]
            }
        }
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
        image = sample['image'].clone().detach().to(self.trainer.device)
        gradient = self.compute_gradient(sample)
        if gradient is None:
            return None
        sign_data_grad = gradient.sign()
        perturbed_img = image + epsilon * gradient * sign_data_grad
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
        return perturbed_img

    def masked_fgsm(self, sample, epsilon):
        image = sample['image'].clone().detach().to(self.trainer.device)
        masked_gradient = self.compute_masked_gradient(sample)
        if masked_gradient is None:
            return None
        # Calculate the sign of the masked gradient
        sign_data_grad = masked_gradient.sign()
        perturbed_img = image + epsilon * masked_gradient * sign_data_grad
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
        return perturbed_img

    def pgd(self, sample, epsilon, alpha, num_iter):
        image = sample['image'].clone().detach().to(self.trainer.device)
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
        image = sample['image'].clone().detach().to(self.trainer.device)
        perturbed_image = image.clone().detach().requires_grad_(True)
        for _ in range(num_iter):
            gradient = self.compute_masked_gradient(sample)
            # Update perturbed image using FGSM-style step
            perturbed_image = perturbed_image + alpha * gradient.sign()
            # Project back to the epsilon-ball and [0, 1] range
            delta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
            perturbed_image = torch.clamp(image + delta, 0, 1).detach().requires_grad_(True)
        return perturbed_image
    
    def loss_total(self, lamda1, lamda2, lamda3, m, x, x_adv, key_filters):
        def loss_l2_perturbation():
            # 假设图像范围是 [-0.5, 0.5] 或 [0, 1]，保持一致性即可
            squared_diff = torch.square((x - x_adv) * m)
            l2_dist_per_sample = torch.sum(squared_diff, dim=[1, 2, 3])  # 对 HWC 维度求和，保留 batch
            loss = torch.sum(l2_dist_per_sample)  # batch 总和，类似原代码中 tf.reduce_sum(self.l2dist)
            return loss
        
        def loss_filter_diff():
            loss = torch.tensor(0.0, device=self.trainer.device)
            for layer_name, channel_idx in key_filters.items():
                if channel_idx:
                    activation_x = activations_x[layer_name][:, channel_idx]  # shape: [B, H, W]
                    activation_x_adv = activations_x_adv[layer_name][:, channel_idx]
                    activation_avg = (activation_x + activation_x_adv) / 2.0
                    l2_norm_squared = torch.sum(activation_avg ** 2, dim=[1, 2])
                    loss += torch.mean(l2_norm_squared)
                    loss = -loss
            return loss
        
        def loss_misclassification():
            loss_fn = v8DetectionLoss(self.trainer.model)
            loss_fn, _ = loss_fn(x_pred, x_adv_pred)
            loss = -loss_fn[1]
            return loss
        
        from analyzer.utils import SaveFeatures
        save_features = SaveFeatures()
        save_features.register_hooks(module=self.trainer.model, parent_path="")
        
        x_pred = self.trainer.model(x)
        activations_x = save_features.get_features()
        x_adv_pred = self.trainer.model(x_adv)
        activations_x_adv = save_features.get_features()
        save_features.close()
        
        loss_tatal = lamda1 * loss_l2_perturbation() + lamda2 * loss_filter_diff() + lamda3 * loss_misclassification()
        
        return loss_tatal

    def filter_attack(self, sample, lr, epsilon, num_iter, key_filters):
        x = sample['image'].clone().detach().to(self.trainer.device)
        m = sample['mask'].clone().detach().to(self.trainer.device)
        x_adv = x.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([x_adv], lr=lr)
        
        for step in range(num_iter):
            optimizer.zero_grad()
            loss = self.loss_total(lamda1=0.1, lamda2=1.0, lamda3=1.0, m=m, x=x, x_adv=x_adv, key_filters=key_filters)
            loss.backward() # type: ignore
            
            optimizer.step()
            with torch.no_grad():
                delta = torch.clamp(x_adv - x, -epsilon, epsilon)
                x_adv = torch.clamp(x + delta, 0, 1).detach().requires_grad_(True)
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
        return x_adv
    
    def single_filter_attack(self, sample, lr, epsilon, num_iter, key_filters):
        x = sample['image'].clone().detach().to(self.trainer.device).unsqueeze(0)
        m = sample['mask'].clone().detach().to(self.trainer.device).unsqueeze(0)
        x_adv = x.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([x_adv], lr=lr)
        
        for step in range(num_iter):
            optimizer.zero_grad()
            loss = self.loss_total(lamda1=0.1, lamda2=1.0, lamda3=1.0, m=m, x=x, x_adv=x_adv, key_filters=key_filters)
            loss.backward() # type: ignore
            
            optimizer.step()
            with torch.no_grad():
                delta = torch.clamp(x_adv - x, -epsilon, epsilon)
                x_adv = torch.clamp(x + delta, 0, 1).detach().requires_grad_(True)
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
        return x_adv
    
    def batch_attack(self, method, output_dir = "./mask-attack/result", **kwargs):
        # Validate the chosen attack method
        if method not in self.method_config:
            raise ValueError(f"Unsupported attack method: {method}. "
                             f"Available methods are: {list(self.method_config.keys())}")                                                                                                                                                                                   
        else:
            try:
                # 动态提取参数并调用攻击函数
                attack_function = self.method_config[method]['attack_function']
                params_name = [para_name for para_name in self.method_config[method]['params_name']]
                params_value = [kwargs[para_name] for para_name in self.method_config[method]['params_name']]
            except KeyError as e:
                raise ValueError(f"Missing required parameter: {e}")
        
        # Create DataLoader for batch processing
        batch_loader_with_progress = tqdm(
            self.batch_loader,
            desc=(
                f"Processing Batches in {method}-"
                f"{'-'.join([f'{params_name[para_idx]}-{params_value[para_idx]:.4f}' for para_idx in range(len(params_name))])}"                         
            ),
            total=len(self.batch_loader)
        )
        for batch in batch_loader_with_progress:
            if batch is None:  # Skip empty batches
                continue
                  
            perturbed_batch = attack_function(batch, *params_value)
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

            

    def adversarial_training(self):
        pass