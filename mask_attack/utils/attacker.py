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
from mask_attack.utils.dataset import CustomDataset, custom_collate_fn
import torch
import warnings

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

        # Process each batch
        for batch in batch_loader_with_progress:
            if batch is None:  # Skip empty batches
                continue
                  
            perturbed_batch = attack_function(batch, *params_value)
            
            # save pertrubed images
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
            # perturbed_np_batch = (perturbed_batch * 255).cpu().clamp(0, 255).byte().numpy()
            # perturbed_np_batch = perturbed_np_batch.transpose(0, 2, 3, 1)  # (N, H, W, C)
            
            # for i in range(len(perturbed_np_batch)):
            #     perturbed_img = perturbed_np_batch[i]
                
            #     original_image_path = batch['image_path'][i]
            #     custom_image_path = os.path.join(
            #         output_dir, "images", os.path.basename(original_image_path)
            #     )
            #     Image.fromarray(perturbed_img, mode='RGB').save(custom_image_path)
                
            #     original_label_path = batch['label_path'][i]
            #     custom_labels_path = os.path.join(
            #         output_dir, "labels", os.path.basename(original_label_path)
            #     )
            #     shutil.copy2(original_label_path, custom_labels_path)
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
    
    
def classes_batch_attack_gtsrb(trainer,
                               test_classes_root,
                               output_root,
                               method='pgd', 
                               epsilon=0.02, 
                               alpha=0.0003, 
                               num_iter=100):
    for class_name in os.listdir(test_classes_root):
        if class_name not in ['0', '2', '14', '23', '39']:
            continue
        
        class_dir = os.path.join(test_classes_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        images_dir_path = os.path.join(class_dir, "images")
        labels_dir_path = os.path.join(class_dir, "labels")

        if not (os.path.exists(images_dir_path) and os.path.exists(labels_dir_path)):
            print(f"Skip class {class_name}: lack images or labels dir")
            continue
        
        epsilon_str = f"{epsilon:.4f}".replace('.', '-')
        alpha_str = f"{alpha:.4f}".replace('.', '-')
        if method == 'fgsm' or method == 'masked_fgsm':
            output_dir = os.path.join(output_root, f"{method}_{epsilon_str}", class_name)
        elif method == 'pgd' or method == 'masked_pgd':
            output_dir = os.path.join(output_root, f"{method}_{epsilon_str}_{alpha_str}_{num_iter}", class_name)
        else:
            output_dir = os.path.join(output_root, class_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            train_dataset = CustomDataset(
                images_dir_path=images_dir_path,
                labels_dir_path=labels_dir_path,
                image_width=640,
                image_height=640
            )

            attacker = Attacker(
                trainer=trainer,
                dataset=train_dataset,
                batch_size=32,
                custom_collate_fn=custom_collate_fn
            )

            # 执行攻击（忽略警告）
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                attacker.batch_attack(method=method, output_dir=output_dir, epsilon=epsilon, alpha=alpha, num_iter=num_iter)

            print(f"Attack class {class_name}: Finished, Save in {output_dir}")

        except Exception as e:
            print(f"Attack class {class_name} Error, {e}")