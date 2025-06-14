import torch
import torch.nn as nn

def loss_total(trainer, lambda1, lambda2, lambda3, m, x, x_adv, key_filters):
    def loss_l2_perturbation():
        # 假设图像范围是 [-0.5, 0.5] 或 [0, 1]，保持一致性即可
        squared_diff = torch.square((x - x_adv) * m)
        l2_dist_per_sample = torch.sum(squared_diff, dim=[1, 2, 3])  # 对 HWC 维度求和，保留 batch
        loss = torch.sum(l2_dist_per_sample)  # batch 总和，类似原代码中 tf.reduce_sum(self.l2dist)
        return loss
    
    def loss_filter_diff():
        loss = torch.tensor(0.0, device=trainer.device)
        for layer_name, channel_idx in key_filters.items():
            if channel_idx:
                activation_x = activations_x[layer_name][:, channel_idx]  # shape: [B, H, W]
                activation_x_adv = activations_x_adv[layer_name][:, channel_idx]
                activation_avg = (activation_x - activation_x_adv) / 2.0
                l2_norm_squared = torch.sum(activation_avg ** 2, dim=[1, 2])
                loss += torch.mean(l2_norm_squared)
        return loss
    
    def loss_misclassification():
        def get_class_logits(model_raw_output, trainer):
            model = trainer.model[-1]
            nc = model.nc
            reg_max_x4 = model.reg_max * 4
            no_total = nc + reg_max_x4
            feats_list = model_raw_output[1] if isinstance(model_raw_output, (tuple, list)) and len(model_raw_output) == 2 else model_raw_output
            batch_s = feats_list[0].shape[0]
            _dist_logits, class_score_logits = torch.cat([xi.view(batch_s, no_total, -1) for xi in feats_list], 2).split((reg_max_x4, nc), 1)
            return class_score_logits.permute(0, 2, 1).contiguous()
        pred_logits_x = get_class_logits(x_pred, trainer.model)
        pred_logits_x_adv = get_class_logits(x_adv_pred, trainer.model)
        # 原始标签（真实类别）作为目标
        target_labels = torch.sigmoid(pred_logits_x).detach()  # 原始输出概率
        # 攻击目标：最大化误分类损失（取负 BCE 损失）
        bce_criterion = nn.BCEWithLogitsLoss(reduction="sum")
        misclassification_loss = bce_criterion(pred_logits_x_adv, target_labels)
        return misclassification_loss
        
    from analyzer.utils import SaveFeatures
    from analyzer.utils import LayerConfig
    
    activations_x = {}
    activations_x_adv = {}
    for config in LayerConfig:
        module_index = config.index
        layer_configs = config.layers
        parent_path = config.name
        # 获取目标模块
        try:
            module = trainer.model.model[module_index]
        except IndexError:
            print(f"[WARNING] Module index {module_index} out of range, skipping '{parent_path}'")
            continue
        # 创建分析器并执行分析
        save_features = SaveFeatures()
        save_features.register_hooks(module=module, parent_path=parent_path)
        with torch.no_grad():
            x_pred = trainer.model(x)
        activations_x.update(save_features.get_features())
        x_adv_pred = trainer.model(x_adv)
        activations_x_adv.update(save_features.get_features())
        save_features.close()
        
    # print("\n=== 特征验证报告 ===")
    # for key in activations_x:
    #     feat_x = activations_x[key]
    #     feat_adv = activations_x_adv[key]
        
    #     print(f"\n层: {key}")
    #     print(f"特征形状 - 原始: {feat_x.shape}, 对抗: {feat_adv.shape}")
    #     print(f"特征差异 - 最大绝对差: {torch.max(torch.abs(feat_x - feat_adv)).item():.4f}")
    #     print(f"特征范围 - 原始: [{feat_x.min():.4f}, {feat_x.max():.4f}], 对抗: [{feat_adv.min():.4f}, {feat_adv.max():.4f}]")
        
    #     # 验证通道差异（选择前3个通道）
    #     for i in range(3):
    #         diff = torch.abs(feat_x[0,i,:,:] - feat_adv[0,i,:,:])
    #         print(f"通道{i}差异 - 平均差异: {torch.mean(diff):.4f}, 最大差异: {torch.max(diff):.4f}")
    
    loss_1 = loss_l2_perturbation()
    loss_2 = loss_filter_diff()
    loss_3 = loss_misclassification()
    
    total_loss = lambda1 * loss_1 - lambda2 * loss_2 - lambda3 * loss_3
    
    return (loss_1, loss_2, loss_3, total_loss)