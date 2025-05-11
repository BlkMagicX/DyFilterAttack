import os
from mask_attack.utils.dataset import CustomDataset, custom_collate_fn
from mask_attack.utils.attacker import Attacker
from mask_attack.utils.dataset import create_temp_yaml
import warnings

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
            

class ValGTSRB():
    def __init__(self, model, attack_result_root = "./mask_attack/attack_result"):
        self.model = model
        self.attack_result_root = attack_result_root
    def single_class_param_val(self, class_name, method='pgd', epsilon=0.02, alpha=0.0003, num_iter=100):
        if class_name not in ['0', '2', '14', '23', '39']:
            return
        
        epsilon_str = f"{epsilon:.4f}".replace('.', '-')
        alpha_str = f"{alpha:.4f}".replace('.', '-')
        if method == 'fgsm' or method == 'masked_fgsm':
            attack_info = f'{method}_{epsilon_str}_{class_name}'
            attack_result_dir = os.path.join(self.attack_result_root, f"{method}_{epsilon_str}", class_name)
        elif method == 'pgd' or method == 'masked_pgd':
            attack_info = f'{method}_{epsilon_str}_{alpha_str}_{num_iter}_{class_name}'
            attack_result_dir = os.path.join(self.attack_result_root, f"{method}_{epsilon_str}_{alpha_str}_{num_iter}", class_name)
        else:
            attack_info = f'{method}_{epsilon_str}_{class_name}'
            attack_result_dir = os.path.join(self.attack_result_root, class_name)
        if not os.path.isdir(attack_result_dir):
            return

        temp_yaml = create_temp_yaml(train_path='../GTSRB/train/images', val_path=attack_result_dir, origin_yaml_path="./mask_attack/data.yaml")            
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                val_metrics = self.model.val(data=temp_yaml, save_json=True)
            print(f"Val {attack_info}: Finished.")

        except Exception as e:
            print(f"Val {attack_info} Error, {e}")