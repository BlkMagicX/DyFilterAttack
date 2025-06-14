import os
import shutil
from PIL import Image
from tqdm import tqdm
from ultralytics.utils.loss import v8DetectionLoss
from DyFilterAttack.mask_attack.utils.loss import loss_total
from torch.utils.data import DataLoader
from DyFilterAttack.mask_attack.utils.CustomDataset import custom_collate_fn
import torch

class Attacker:
    def __init__(self, trainer, dataset, batch_size=1, custom_collate_fn=custom_collate_fn):
        """
        Initialize the Attacker
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
                'params_name': ["lr", "epsilon", "num_iter", "lambda1", "lambda2", "lambda3", "key_filters"]
            }
        }
    
    ### NEW HELPER FUNCTION ###
    def _format_gt_for_loss(self, sample):
        """
        Formats the ground truth from a padded batch into the 2D tensor format
        expected by ultralytics v8DetectionLoss.
        """
        # Padded tensors from the dataloader
        padded_batch_indices = sample['batch_indices']  # Shape: (B, N_max, 1)
        padded_classes = sample['classes']              # Shape: (B, N_max, 1)
        padded_bboxes = sample['bboxes']                # Shape: (B, N_max, 4)

        # Create a boolean mask to filter out padded values (-1)
        valid_mask = (padded_classes.view(-1) >= 0)

        # Flatten the 3D tensors to 2D and then apply the mask
        gt_batch_idx = padded_batch_indices.view(-1, 1)[valid_mask]
        gt_classes = padded_classes.view(-1, 1)[valid_mask]
        gt_bboxes = padded_bboxes.view(-1, 4)[valid_mask]

        return {
            'batch_idx': gt_batch_idx,
            'cls': gt_classes,
            'bboxes': gt_bboxes,
        }

    ### MODIFIED ###
    def compute_gradient(self, sample):
        image = sample['image']
        # Ensure gradient computation is enabled for the input image
        image.requires_grad = True

        pred = self.trainer.model(image)
        
        # Format the ground truth to be compatible with the loss function
        gt_dict = self._format_gt_for_loss(sample)

        loss_fn = v8DetectionLoss(self.trainer.model)
        loss, _ = loss_fn(pred, gt_dict)
        self.trainer.model.zero_grad()
        loss = loss.sum()
        loss.backward()
        
        # Clone the gradient to avoid issues with subsequent operations
        gradient = image.grad.data.clone()
        image.grad = None  # Clear gradient from the image tensor
        image.requires_grad = False
        
        return gradient
        
    ### MODIFIED ###
    def compute_masked_gradient(self, sample):
        image = sample['image']
        mask = sample['mask']
        image.requires_grad = True

        pred = self.trainer.model(image)
        
        # Format the ground truth
        gt_dict = self._format_gt_for_loss(sample)

        loss_fn = v8DetectionLoss(self.trainer.model)
        loss, _ = loss_fn(pred, gt_dict)
        self.trainer.model.zero_grad()
        loss = loss.sum()
        loss.backward()
        
        gradient = image.grad.data.clone()
        masked_gradient = gradient * mask
        image.grad = None # Clear gradient from the image tensor
        image.requires_grad = False
        
        return masked_gradient
        
    ### MODIFIED (Standard FGSM Implementation) ###
    def fgsm(self, sample, epsilon):
        image = sample['image'].clone().detach()
        gradient = self.compute_gradient(sample)
        if gradient is None:
            return None
        
        # Standard FGSM adds perturbation based on the sign of the gradient
        sign_data_grad = gradient.sign()
        perturbed_img = image + epsilon * sign_data_grad
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
        return perturbed_img

    ### MODIFIED (Standard FGSM Implementation) ###
    def masked_fgsm(self, sample, epsilon):
        image = sample['image'].clone().detach()
        masked_gradient = self.compute_masked_gradient(sample)
        if masked_gradient is None:
            return None
            
        sign_data_grad = masked_gradient.sign()
        perturbed_img = image + epsilon * sign_data_grad
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
        return perturbed_img

    ### MODIFIED (Corrected PGD Loop Logic) ###
    def pgd(self, sample, epsilon, alpha, num_iter):
        # Create a copy of the sample dictionary to avoid modifying the original batch data
        current_sample = sample.copy()
        original_image = current_sample['image'].clone().detach()
        perturbed_image = original_image.clone().detach()

        for _ in range(num_iter):
            # The gradient needs to be computed on the *current perturbed image*
            current_sample['image'] = perturbed_image.clone().requires_grad_(True)
            
            gradient = self.compute_gradient(current_sample)
            
            with torch.no_grad():
                # Update perturbed image using the gradient sign
                perturbed_image = perturbed_image + alpha * gradient.sign()
                # Project the perturbation back to the epsilon-ball around the ORIGINAL image
                delta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
                # Clip the final image to the valid [0, 1] range
                perturbed_image = torch.clamp(original_image + delta, 0, 1)
        
        return perturbed_image.detach()
    
    ### MODIFIED (Corrected PGD Loop Logic) ###
    def masked_pgd(self, sample, epsilon, alpha, num_iter):
        current_sample = sample.copy()
        original_image = current_sample['image'].clone().detach()
        perturbed_image = original_image.clone().detach()

        for _ in range(num_iter):
            # The gradient needs to be computed on the *current perturbed image*
            current_sample['image'] = perturbed_image.clone().requires_grad_(True)

            gradient = self.compute_masked_gradient(current_sample)
            
            with torch.no_grad():
                perturbed_image = perturbed_image + alpha * gradient.sign()
                delta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
                perturbed_image = torch.clamp(original_image + delta, 0, 1)
        
        return perturbed_image.detach()
        
    def filter_attack(self, sample, lr, epsilon, num_iter, lambda1, lambda2, lambda3, key_filters):
        # ... (Your existing filter_attack code remains unchanged) ...
        # NOTE: Ensure loss_total correctly handles batched or single inputs based on this logic
        if len(sample['image'].size()) == 3:
            x = sample['image'].clone().detach().to(self.trainer.device).unsqueeze(0).requires_grad_(True)
            m = sample['mask'].clone().detach().to(self.trainer.device).unsqueeze(0)
        elif len(sample['image'].size()) == 4:
            x = sample['image'].clone().detach().to(self.trainer.device).requires_grad_(True)
            m = sample['mask'].clone().detach().to(self.trainer.device)
        else:
            raise ValueError("Wrong input image dimensions")

        random_noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = (x + random_noise).clamp(0, 1).detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([x_adv], lr=lr)
        
        print(f"{'Loss:':>9}"
              f"{'loss_total':>16}"
              f"{'loss_1':>16}"
              f"{'loss_2':>16}"
              f"{'loss_3':>16}")
        
        for step in range(num_iter):
            optimizer.zero_grad()
            
            # Here you might need to handle gt format for your custom loss_total as well
            # depending on its implementation.
            loss = loss_total(self.trainer, 
                              lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, 
                              m=m, x=x, x_adv=x_adv, key_filters=key_filters)
            total_loss = loss[-1]
            total_loss.backward()
            
            with torch.no_grad():
                if x_adv.grad is not None:
                    x_adv.grad *= m
            
            torch.nn.utils.clip_grad_norm_(x_adv, max_norm=1.0)
            
            optimizer.step()
            with torch.no_grad():
                delta = torch.clamp(x_adv.data - x.data, -epsilon, epsilon)
                x_adv.data = torch.clamp(x.data + delta, 0, 1)
                optimizer.param_groups[0]['params'][0].data = x_adv
            
                if step % 10 == 0:
                    print(f"[{step:>3}/{num_iter:>3}]"
                          f"{loss[3].item():>16.6f}"
                          f"{loss[0].item():>16.6f}"
                          f"{loss[1].item():>16.6f}"
                          f"{loss[2].item():>16.6f}")
        
        return x_adv

    def batch_attack(self, method, output_dir = "./mask-attack/result", **kwargs):
        # ... (Your existing batch_attack code remains unchanged) ...
        if method not in self.method_config:
            raise ValueError(f"Unsupported attack method: {method}. "
                             f"Available methods are: {list(self.method_config.keys())}")                                                                                                                                                                                                                                                                                                                                                        
        else:
            try:
                attack_function = self.method_config[method]['attack_function']
                params_name = self.method_config[method]['params_name']
                params_value = [kwargs[para_name] for para_name in params_name]
            except KeyError as e:
                raise ValueError(f"Missing required parameter: {e}")
        
        param_descriptions = []
        # Create a zip of names and values, then iterate
        for para_name, para_value in zip(params_name, params_value):
            if para_name == "key_filters":
                pass # Or add custom description
            elif isinstance(para_value, (int, float)):
                param_descriptions.append(f"{para_name}-{para_value:.4f}")
            else:
                param_descriptions.append(f"{para_name}-{para_value}")
        
        desc = f"Processing Batches in {method}-{'-'.join(param_descriptions)}"
        batch_loader_with_progress = tqdm(self.batch_loader, desc=desc, total=len(self.batch_loader))
        for batch in batch_loader_with_progress:
            if batch is None:
                continue
                
            perturbed_batch = attack_function(batch, *params_value)
            
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
            for i in range(len(perturbed_batch)):
                perturbed_img = perturbed_batch[i].cpu()
                original_image_path = batch['image_path'][i]
                original_label_path = batch['label_path'][i]

                perturbed_np = (perturbed_img * 255).clamp(0, 255).byte().numpy().transpose(1, 2, 0)
                custom_image_path = os.path.join(output_dir, "images", os.path.basename(original_image_path))
                Image.fromarray(perturbed_np).save(custom_image_path)
                
                custom_labels_path = os.path.join(output_dir, "labels", os.path.basename(original_label_path))
                shutil.copy2(original_label_path, custom_labels_path)