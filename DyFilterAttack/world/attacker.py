import torch

import forward
import loss
import weight


class WorldAttacker:
    def __init__(self, model_path):
        self.world, self.layers = forward.setup_model(model_path, verbose=True)

    def choose_attack_layer(self, attack_layer_name):
        self.attack_layer_name = attack_layer_name
        self.attack_layer = self.layers["attack_layer_name"]

    def forward(self, image_path, target_layer_name='model.model.22.cv2'):
        # ! 这里暂时是 image_path, 后面要修改成image_adv的tensor向量
        self.image_tensor = forward.preprocess_image(image_path, self.world.device)
        # compute activation and gard
        self.grad_orig, self.grad_target, self.activation = forward.compute_gradients_y_det_and_activation(
            world=self.world, image_tensor=self.image_tensor, target_layer_name=target_layer_name
        )
        # compute parameters
        self.alpha_o, self.alpha_t = weight.compute_basic_weights(grad_orig=self.grad_orig, grad_target=self.grad_target)
        self.spatial_mask = weight.compute_spatial_mask(grad_orig=self.grad_orig)
        self.direction = weight.compute_direction(alpha_o=self.alpha_o, alpha_t=self.alpha_t)
        self.weight_p, self.weight_n = weight.compute_full_weight(alpha_o=self.alpha_o, alpha_t=self.alpha_t, direction=self.direction)
        self.mask = weight.compute_mask(weight_p=self.weight_p, weight_n=self.weight_n, spatial_mask=self.spatial_mask)
