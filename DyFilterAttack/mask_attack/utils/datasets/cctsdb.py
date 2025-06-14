import os
from DyFilterAttack.mask_attack.utils.CustomDataset import CustomDataset, custom_collate_fn
from DyFilterAttack.mask_attack.utils.attacker import Attacker
from DyFilterAttack.mask_attack.utils.CustomDataset import create_temp_yaml
import warnings

def build_attack_info(method, **kwargs):
    if method in ['fgsm', 'masked_fgsm']:
        epsilon = f"{kwargs.get('epsilon', 0.02):.4f}".replace('.', '-')
        return f"{method}_{epsilon}"
    elif method in ['pgd', 'masked_pgd']:
        epsilon = f"{kwargs.get('epsilon', 0.02):.4f}".replace('.', '-')
        alpha = f"{kwargs.get('alpha', 0.0003):.4f}".replace('.', '-')
        num_iter = kwargs.get('num_iter', 100)
        return f"{method}_{epsilon}_{alpha}_{num_iter}"
    elif method == 'filter_attack':
        epsilon = f"{kwargs.get('epsilon', 0.02):.4f}".replace('.', '-')
        lr = f"{kwargs.get('lr', 0.001):.4f}".replace('.', '-')
        num_iter = kwargs.get('num_iter', 100)
        lambda1 = kwargs.get('lambda1', 1)
        lambda2 = kwargs.get('lambda2', 0.1)
        lambda3 = kwargs.get('lambda3', 1)
        return f"{method}_{epsilon}_{lr}_{num_iter}_{lambda1}_{lambda2}_{lambda3}"
    else:
        return method


def batch_attack_cctsdb(trainer,
                        classes_name,
                        batch_size,
                        test_classes_root,
                        output_root,
                        method,
                        **kwargs):

    if classes_name:
        for class_name in os.listdir(test_classes_root):
            if class_name not in classes_name:
                continue

            class_dir = os.path.join(test_classes_root, class_name)
            if not os.path.isdir(class_dir):
                continue

            images_dir_path = os.path.join(class_dir, "images")
            labels_dir_path = os.path.join(class_dir, "labels")

            if not (os.path.exists(images_dir_path) and os.path.exists(labels_dir_path)):
                print(f"Skip class {class_name}: lack images or labels dir")
                continue

            # 构建输出路径
            attack_info = build_attack_info(method, **kwargs)
            output_dir = os.path.join(output_root, attack_info, class_name)
            os.makedirs(output_dir, exist_ok=True)
    else:
        images_dir_path = os.path.join(test_classes_root, "images")
        labels_dir_path = os.path.join(test_classes_root, "labels")

        if not (os.path.exists(images_dir_path) and os.path.exists(labels_dir_path)):
            print(f"Can not find dir: {images_dir_path} or {labels_dir_path}")
            return 
        
        # 构建输出路径
        attack_info = build_attack_info(method, **kwargs)
        output_dir = os.path.join(output_root, attack_info)
        os.makedirs(output_dir, exist_ok=True)
    
    train_dataset = CustomDataset(
        images_dir_path=images_dir_path,
        labels_dir_path=labels_dir_path,
        image_width=640,
        image_height=640
    )

    attacker = Attacker(
        trainer=trainer,
        dataset=train_dataset,
        batch_size=batch_size,
        custom_collate_fn=custom_collate_fn
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        attacker.batch_attack(method=method, output_dir=output_dir, **kwargs)

    print(f"Attack class {classes_name}: Finished, Save in {output_dir}")
    
    # merge_dataset_structure(output_root, os.path.join(os.path.dirname(output_root), 'no_classes_result'))
