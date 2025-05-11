import os
import warnings
from mask_attack.utils.dataset import create_temp_yaml

class Val_GTSRB():
    def __init__(self, model, attack_result_root = "./mask_attack/attack_result"):
        self.model = model
        self.attack_result_root = attack_result_root
    def single_class_para_val(self, class_name, method='pgd', epsilon=0.02, alpha=0.0003, num_iter=100):
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
            
    def single_class_val(self, class_name):
        self.single_class_para_val(class_name, "fgsm", 0.5)
        self.single_class_para_val(class_name, "fgsm", 0.4)
        self.single_class_para_val(class_name, "fgsm", 0.3)
        self.single_class_para_val(class_name, "fgsm", 0.2)
        self.single_class_para_val(class_name, "fgsm", 0.1)
        self.single_class_para_val(class_name, "fgsm", 0.08)
        self.single_class_para_val(class_name, "fgsm", 0.05)
        self.single_class_para_val(class_name, "fgsm", 0.03)
        self.single_class_para_val(class_name, "fgsm", 0.01)
        self.single_class_para_val(class_name, "fgsm", 0.007)
        self.single_class_para_val(class_name, "fgsm", 0.005)
        self.single_class_para_val(class_name, "fgsm", 0.003)
        self.single_class_para_val(class_name, "fgsm", 0.001)
        self.single_class_para_val(class_name, "masked_fgsm", 0.5)
        self.single_class_para_val(class_name, "masked_fgsm", 0.4)
        self.single_class_para_val(class_name, "masked_fgsm", 0.3)
        self.single_class_para_val(class_name, "masked_fgsm", 0.2)
        self.single_class_para_val(class_name, "masked_fgsm", 0.1)
        self.single_class_para_val(class_name, "masked_fgsm", 0.08)
        self.single_class_para_val(class_name, "masked_fgsm", 0.05)
        self.single_class_para_val(class_name, "masked_fgsm", 0.03)
        self.single_class_para_val(class_name, "masked_fgsm", 0.01)
        self.single_class_para_val(class_name, "masked_fgsm", 0.007)
        self.single_class_para_val(class_name, "masked_fgsm", 0.005)
        self.single_class_para_val(class_name, "masked_fgsm", 0.003)
        self.single_class_para_val(class_name, "masked_fgsm", 0.001)
        self.single_class_para_val(class_name, "pgd", 0.05, 0.0005, 150)
        self.single_class_para_val(class_name, "pgd", 0.05, 0.0005, 100)
        self.single_class_para_val(class_name, "pgd", 0.05, 0.0008, 100)
        self.single_class_para_val(class_name, "pgd", 0.05, 0.0015, 50)
        self.single_class_para_val(class_name, "pgd", 0.03, 0.0003, 150)
        self.single_class_para_val(class_name, "pgd", 0.03, 0.0003, 100)
        self.single_class_para_val(class_name, "pgd", 0.03, 0.0005, 100)
        self.single_class_para_val(class_name, "pgd", 0.03, 0.0008, 50)
        self.single_class_para_val(class_name, "pgd", 0.02, 0.0002, 150)
        self.single_class_para_val(class_name, "pgd", 0.02, 0.0002, 100)
        self.single_class_para_val(class_name, "pgd", 0.02, 0.0003, 100)
        self.single_class_para_val(class_name, "pgd", 0.02, 0.0006, 50)
        self.single_class_para_val(class_name, "pgd", 0.01, 0.0001, 150)
        self.single_class_para_val(class_name, "pgd", 0.01, 0.0001, 100)
        self.single_class_para_val(class_name, "pgd", 0.01, 0.00015, 100)
        self.single_class_para_val(class_name, "pgd", 0.01, 0.0003, 50)
        self.single_class_para_val(class_name, "pgd", 0.008, 0.00008, 150)
        self.single_class_para_val(class_name, "pgd", 0.008, 0.00008, 100)
        self.single_class_para_val(class_name, "pgd", 0.008, 0.00010, 100)
        self.single_class_para_val(class_name, "pgd", 0.008, 0.00020, 50)
        self.single_class_para_val(class_name, "pgd", 0.005, 0.00005, 150)
        self.single_class_para_val(class_name, "pgd", 0.005, 0.00005, 100)
        self.single_class_para_val(class_name, "pgd", 0.005, 0.00008, 100)
        self.single_class_para_val(class_name, "pgd", 0.005, 0.00015, 50)
        self.single_class_para_val(class_name, "pgd", 0.003, 0.00003, 150)
        self.single_class_para_val(class_name, "pgd", 0.003, 0.00003, 100)
        self.single_class_para_val(class_name, "pgd", 0.003, 0.00005, 100)
        self.single_class_para_val(class_name, "pgd", 0.003, 0.00008, 50)
        self.single_class_para_val(class_name, "pgd", 0.002, 0.00002, 150)
        self.single_class_para_val(class_name, "pgd", 0.002, 0.00002, 100)
        self.single_class_para_val(class_name, "pgd", 0.002, 0.00003, 100)
        self.single_class_para_val(class_name, "pgd", 0.002, 0.00006, 50)
        self.single_class_para_val(class_name, "pgd", 0.001, 0.00001, 150)
        self.single_class_para_val(class_name, "pgd", 0.001, 0.00001, 100)
        self.single_class_para_val(class_name, "pgd", 0.001, 0.000015, 100)
        self.single_class_para_val(class_name, "pgd", 0.001, 0.00003, 50)
        self.single_class_para_val(class_name, "masked_pgd", 0.05, 0.0005, 150)
        self.single_class_para_val(class_name, "masked_pgd", 0.05, 0.0005, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.05, 0.0008, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.05, 0.0015, 50)
        self.single_class_para_val(class_name, "masked_pgd", 0.03, 0.0003, 150)
        self.single_class_para_val(class_name, "masked_pgd", 0.03, 0.0003, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.03, 0.0005, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.03, 0.0008, 50)
        self.single_class_para_val(class_name, "masked_pgd", 0.02, 0.0002, 150)
        self.single_class_para_val(class_name, "masked_pgd", 0.02, 0.0002, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.02, 0.0003, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.02, 0.0006, 50)
        self.single_class_para_val(class_name, "masked_pgd", 0.01, 0.0001, 150)
        self.single_class_para_val(class_name, "masked_pgd", 0.01, 0.0001, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.01, 0.00015, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.01, 0.0003, 50)
        self.single_class_para_val(class_name, "masked_pgd", 0.008, 0.00008, 150)
        self.single_class_para_val(class_name, "masked_pgd", 0.008, 0.00008, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.008, 0.00010, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.008, 0.00020, 50)
        self.single_class_para_val(class_name, "masked_pgd", 0.005, 0.00005, 150)
        self.single_class_para_val(class_name, "masked_pgd", 0.005, 0.00005, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.005, 0.00008, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.005, 0.00015, 50)
        self.single_class_para_val(class_name, "masked_pgd", 0.003, 0.00003, 150)
        self.single_class_para_val(class_name, "masked_pgd", 0.003, 0.00003, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.003, 0.00005, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.003, 0.00008, 50)
        self.single_class_para_val(class_name, "masked_pgd", 0.002, 0.00002, 150)
        self.single_class_para_val(class_name, "masked_pgd", 0.002, 0.00002, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.002, 0.00003, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.002, 0.00006, 50)
        self.single_class_para_val(class_name, "masked_pgd", 0.001, 0.00001, 150)
        self.single_class_para_val(class_name, "masked_pgd", 0.001, 0.00001, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.001, 0.000015, 100)
        self.single_class_para_val(class_name, "masked_pgd", 0.001, 0.00003, 50)