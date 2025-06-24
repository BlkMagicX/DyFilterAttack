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

    def forward(self, image_path):
        # ! 这里暂时是 image_path, 后面要修改成image_adv的tensor向量
        # save activation
        self.activation = forward.extract_features(self.world, self.layers, image_path, extract_module_name=self.attack_layer_name)
        # extract y_det_orig and y_det_target 
        self.y_det_orig, self.y_det_target = forward.extract_world_y_det(self.world, self.layers)
        # compute 
        self.alpha = weight.compute_alpha(self.activation, self.y_det_orig, self.y_det_target)
        self.spatial_mask = weight.compute_spatial_mask(self.activation, self.y_det_orig)
        self.direction = weight.compute_direction(self.alpha)
        self.weight = weight.compute_weight(self.alpha, self.direction)
        self.weight_p, self.weight_n = weight.decouple_weight_mask(self.alpha, self.direction)

    def 
