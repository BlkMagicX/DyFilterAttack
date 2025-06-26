from ultralytics import YOLOWorld
from pathlib import Path
from itertools import product
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import os
import torch
import shutil
import warnings


import experiment
import forward
import weight
import loss
import dataset


class WorldAttacker:

    def __init__(self, model_path, attack_layer_name, meta_json, images_dir_path):
        self.experiment = self.setup_experiment()
        self.attack_dataset = self.setup_dataset(meta_json, images_dir_path)
        self.world, self.layers = self.setup_model(model_path, verbose=True)
        self.predictor = self.world._smart_load("predictor")(_callbacks=self.world.callbacks)
        self.predictor.setup_model(model=self.world.model, verbose=False)
        self.attack_layer_name, self.attack_layer = self.choose_attack_layer(attack_layer_name)

    def setup_experiment(self):
        base_hyp = dict(eps=0.05, lr=0.005, num_iter=10)
        target_layers = ["model.model.22.cv2", "model.model.19.cv1"]
        optimizer_names = ["adam", "sgd", "adamw"]
        exps = {}
        # create baseline exp
        exps["baseline"] = experiment.ExpConfig(name="baseline", hyp=base_hyp.copy(), target_layer="model.model.22.cv2", optim_name="adam")
        # create exps list
        for layer, opt in product(target_layers, optimizer_names):
            name = f"{layer.split('.')[-1]}_{opt}"
            if name in exps:
                continue
            exps[name] = experiment.ExpConfig(name=name, hyp=base_hyp.copy(), target_layer=layer, optim_name=opt)
        return exps

    def setup_dataset(self, meta_json, images_dir_path):
        attack_dataset = dataset.AttackDataset(meta_json=meta_json, images_dir_path=images_dir_path)
        return attack_dataset

    def setup_model(self, model_path, verbose=True):
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
        else:
            raise ValueError(f"model_path must contain '-world' and suffix .pt/.yaml/.yml, got {model_path}")

        return world, layers

    def choose_attack_layer(self, attack_layer_name):
        attack_layer = self.layers[attack_layer_name]
        return attack_layer_name, attack_layer

    def forward(self, preprocess_tensor, det_ids_batch):
        self.image_tensor = preprocess_tensor.requires_grad_(True)
        # compute activation and gard
        self.grad_orig, self.grad_target, self.activation = forward.compute_gradients_y_det_and_activation(
            world=self.world,
            image_tensor=self.image_tensor,
            target_layer_name=self.attack_layer_name,
            det_ids_batch=det_ids_batch,
            predictor=self.predictor,
        )
        # compute parameters
        self.alpha_o, self.alpha_t = weight.compute_basic_weights(grad_orig=self.grad_orig, grad_target=self.grad_target)
        self.spatial_mask = weight.compute_spatial_mask(grad_orig=self.grad_orig)
        self.direction = weight.compute_direction(alpha_o=self.alpha_o, alpha_t=self.alpha_t)
        self.weight_p, self.weight_n = weight.compute_full_weight(alpha_o=self.alpha_o, alpha_t=self.alpha_t, direction=self.direction)
        self.mask_p, self.mask_n = weight.compute_decouple_mask(weight_p=self.weight_p, weight_n=self.weight_n, spatial_mask=self.spatial_mask)
        self.mask = weight.compute_mask(mask_p=self.mask_p, mask_n=self.mask_n)

    def backward(self, loss, x):
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()
        return None if x.grad is None else x.grad.detach()

    def adversarial_attack(self, batch, experiment):
        hyp = experiment.hyp
        x_orig = batch["image"].to(self.world.device)
        if x_orig.dim() == 3:
            x_orig = x_orig.unsqueeze(0)
        x_orig = x_orig.detach()

        delta = torch.empty_like(x_orig).uniform_(-hyp["eps"], hyp["eps"]).to(self.world.device)
        delta.requires_grad_(True)

        optimizer = experiment.build_optimizer(delta)

        print("start attack for 1 batch")
        print(f"{'Loss:':>9}" f"{'loss_total':>16}" f"{'loss_p':>16}" f"{'loss_s':>16}")
        for step in range(hyp["num_iter"]):
            optimizer.zero_grad()

            x_adv = (x_orig + delta).clamp(0, 1)

            self.forward(x_adv, batch["det_ids"])

            loss_total, loss_promote, loss_surpress = loss.total(
                mask_p=self.mask_p, mask_n=self.mask_n, activation=self.activation, lambda_promote=1.0, lambda_surpress=1.0
            )

            self.backward(loss=loss_total, x=delta)

            assert delta.grad is not None and delta.grad.abs().sum() > 0
            torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                delta.data.clamp_(-hyp["eps"], hyp["eps"])

            if step % 10 == 0:
                print(f"[{step:03d}/{hyp['num_iter']}] " f"loss={loss_total.item():.4f} lp={loss_promote.item():.4f} ls={loss_surpress.item():.4f}")

        return x_adv

    def batch_attack(self, experiment_name, output_dir="./DyFilterAttack/world/result", batch_size=1):
        dataloader = DataLoader(self.attack_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.attack_collate)
        batch_loader_with_progress = tqdm(dataloader, desc=f"Processing Batches: ", total=len(dataloader))
        for batch in batch_loader_with_progress:
            if batch is None:
                continue

            perturbed_batch = self.adversarial_attack(batch, self.experiment[experiment_name])

            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
            for i in range(len(perturbed_batch)):
                perturbed_img = perturbed_batch[i].cpu()
                original_image_path = batch["image_path"][i]
                original_label_path = batch["label_path"][i]

                perturbed_np = (perturbed_img * 255).clamp(0, 255).byte().numpy().transpose(1, 2, 0)
                custom_image_path = os.path.join(output_dir, "images", os.path.basename(original_image_path))

                Image.fromarray(perturbed_np).save(custom_image_path)
                custom_labels_path = os.path.join(output_dir, "labels", os.path.basename(original_label_path))
                shutil.copy2(original_label_path, custom_labels_path)

    def attack_success(self):
        pass

    def evaluation(self):
        pass
