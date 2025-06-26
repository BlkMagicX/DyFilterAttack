import torch
import warnings
from pathlib import Path
from ultralytics import YOLOWorld
from itertools import product
from experiment import ExpConfig


import forward
import weight
import loss


class WorldAttacker:
    def __init__(self, model_path, attack_layer_name):
        self.experiment = self.setup_experiment()
        self.dataset = self.setup_dataset()
        self.world, self.layers = self.setup_model(model_path, verbose=True)
        self.attack_layer_name, self.attack_layer = self.choose_attack_layer(attack_layer_name)

    def setup_experiment(self):
        base_hyp = dict(eps=8 / 255, lr=1e-2, num_iter=40)
        target_layers = ["model.model.22.cv2", "model.model.19.cv1"]
        optimizer_names = ["adam", "sgd", "adamw"]
        exps = {}
        # create baseline exp
        exps["baseline"] = ExpConfig(name="baseline_adam", hyp=base_hyp.copy(), target_layer="model.model.22.cv2", optim_name="adam")
        # create exps list
        for layer, opt in product(target_layers, optimizer_names):
            name = f"{layer.split('.')[-1]}_{opt}"
            if name in exps:
                continue
            exps[name] = ExpConfig(name=name, hyp=base_hyp.copy(), target_layer=layer, optim_name=opt)
        return exps

    def setup_dataset(self):
        dataset = None
        return dataset

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

    def forward(self, preprocessed_image_tensor):
        # ! 这里的image_tensor必须是预处理过后的
        self.image_tensor = preprocessed_image_tensor.requires_grad_(True)
        # compute activation and gard
        self.grad_orig, self.grad_target, self.activation = forward.compute_gradients_y_det_and_activation(
            world=self.world, image_tensor=self.image_tensor, target_layer_name=self.attack_layer_name
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

    def attack_success(self):
        pass

    def adversarial_attack(self, sample, expriment):
        hyp = expriment.hyp
        x_orig = sample["image"].to(self.world.device)
        if x_orig.dim() == 3:
            x_orig = x_orig.unsqueeze(0)
        x_orig = x_orig.detach()

        delta = torch.empty_like(x_orig).uniform_(-hyp["eps"], hyp["eps"]).to(self.world.device)
        delta.requires_grad_(True)

        optimizer = expriment.build_optimizer(delta)

        print("start attack for 1 batch")
        print(f"{'Loss:':>9}" f"{'loss_total':>16}" f"{'loss_p':>16}" f"{'loss_s':>16}")
        for step in range(hyp["num_iter"]):
            optimizer.zero_grad()

            x_adv = (x_orig + delta).clamp(0, 1)

            self.forward(x_adv)

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

    def batch_attack(self):
        pass

    def evaluation(self):
        pass
