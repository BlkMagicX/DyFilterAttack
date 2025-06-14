import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from functools import partial
import torchvision.transforms as T
import os
import glob
from enum import Enum
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import skew
from kneed import KneeLocator  # 自动检测肘点
import numpy as np

class SaveFeatures:
    def __init__(self) -> None:
        self.features = {}
        self.hooks = []
    def hook_fn(self, module, input, output, path) -> None:
        self.features[path] = output
    def register_hooks(self, module, parent_path):
        for name, child in module.named_children():
            current_path = f"{parent_path}.{name}" if parent_path else name
            # if isinstance(child, torch.nn.BatchNorm2d):
            # print(f"Registering Hook: {current_path})"
            hook = child.register_forward_hook(
                hook=partial(self.hook_fn, path=current_path)
            )
            self.hooks.append(hook)
            # 递归调用时保持 parent_path 不变
            self.register_hooks(child, current_path)
    def get_features(self):
        return self.features
    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        
class FilterAnalyzer():
    def __init__(self, model, module, parent_path, origin_img_dir, attack_img_dir):
        self.model = model
        self.module = module
        self.parent_path = parent_path
        self.origin_img_dir = origin_img_dir
        self.attack_img_dir = attack_img_dir
        self.save_features = SaveFeatures()
        self.save_features.register_hooks(module=self.module, parent_path=self.parent_path)
        self.activate_results = None
        
    def _get_yolo_transforms(self, sz: int):
        val_tfms = T.Compose(transforms=[
            T.ToTensor(),
            T.Resize(size=(sz, sz)),
        ])
        train_tfms = val_tfms
        return train_tfms, val_tfms
    
    def _denorm(self, tensor: torch.Tensor) -> np.ndarray:
        """反归一化：将 [0,1] 张量恢复为 [0,255] 图像（HWC）"""
        img = tensor.permute(1, 2, 0).cpu().numpy() * 255
        return np.clip(img, 0, 255).astype(np.uint8)
    
    # def visualize_opt(self, layer, filter, lr=0.1, opt_steps=20, blur=None):
    def visualize_opt(self, target_layer_path, size=56, upscaling_steps=12, upscaling_factor=1.2,
                      filter_idx=0, lr=0.1, opt_steps=20, blur=None, cmd=False) -> None:   
        loss = 0
        sz = size  
        img = np.float32(np.random.uniform(low=150, high=180, size=(sz, sz, 3))/255)
        train_tfms, val_tfms = self._get_yolo_transforms(sz=sz)
        img_tensor = val_tfms(img=img).unsqueeze(0)  # type: ignore # [1, 3, sz, sz]
        img_var = torch.tensor(data=img_tensor, requires_grad=True, device=self.model.device, dtype=torch.float32)
        for step in range(upscaling_steps):
            # print(f"upscaling step{step+1}")
            # 步骤3：初始化优化器（优化图像像素值）
            optimizer = torch.optim.Adam(params=[img_var], lr=lr, weight_decay=1e-6)
            # 步骤4：优化像素值（最大化目标 filter 的激活）
            for n in range(opt_steps):
                optimizer.zero_grad()
                _ = self.model.model(img_var)  # YOLOv8 前向传播（输出可能包含多个特征图，但钩子仅捕获目标层）
                activations = self.save_features.get_features()
                target_features = activations[target_layer_path]
                if target_features is None:
                    raise ValueError(f"未找到目标层特征，路径：{target_layer_path}")
                loss = -target_features[0, filter_idx].mean()  # [0, filter_idx] 对应 batch 0，第 filter_idx 个通道
                loss.backward()
                optimizer.step()
            if cmd:
                print(f"loss in epoch{step+1}: {loss}")
            # 步骤5：反归一化并恢复图像（HWC, [0,255]）
            img = self._denorm(img_var.detach().squeeze(0))  # [3, sz, sz] → [sz, sz, 3]
            # print(img)
            self.output = img
            # 步骤6：上采样图像（为下一轮优化准备更大尺寸）
            sz = int(upscaling_factor * sz)
            img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)  # 上采样
            if blur is not None:
                img = cv2.blur(img, (blur, blur))  # 模糊去高频噪声
            # plt.imshow(X=np.clip(a=self.output, a_min=0, a_max=1))
        file_name = "./filter/"+str(object=target_layer_path)+"_filter_"+str(object=filter_idx+1)+".jpg"
        print(f"save: {file_name}")
        plt.imsave(fname="./filter/layer_"+str(object=target_layer_path)+"_filter_"+str(object=filter_idx+1)+".jpg", arr=self.output)
        if cmd:  
            plt.imshow(X=self.output)
            
    def compute_single_layer_mean_activation(self, img_path, target_layer_path, left_filter_idx, right_filter_idx , size=640):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(cv2.resize(img, (size, size))/255)
    
        # 2. 应用YOLO的数据增强（验证模式）
        _, val_tfms = self._get_yolo_transforms(sz=size)
        img_tensor = val_tfms(img=img).unsqueeze(0)  # [1, 3, sz, sz]
        img_var = img_tensor.detach().clone().requires_grad_(True).to(device=self.model.device, dtype=torch.float32)
        
        with torch.no_grad():
            _ = self.model.model(img_var)  # YOLOv8 前向传播（输出可能包含多个特征图，但钩子仅捕获目标层）
            activations = self.save_features.get_features()
            target_features = activations[target_layer_path][0, left_filter_idx:right_filter_idx, :, :] 
            # mean_activation = np.float32(target_features.mean(axis=[1, 2]).cpu().detach().numpy())
            h, w = target_features.size(0), target_features.size(1)
            l2_norm = torch.norm(target_features, p=2, dim=(1, 2), keepdim=False) / (h * w) # type: ignore # 在空间维度计算 L2 范数
            l2_norm_np = l2_norm.cpu().numpy().astype(np.float32)   # 转换为 NumPy 数组
        
        # mean_activation = torch.from_numpy(mean_activation).to(device=self.model.device, dtype=torch.float32)
        mean_activation = torch.from_numpy(l2_norm_np).to(device=self.model.device, dtype=torch.float32)
        
        return mean_activation

    def multi_img_single_layer_mean_activations(self, target_layer_path, left_filter_idx, right_filter_idx, show=False, size=640):
        def process_folder(sample_num, folder):
            img_paths = glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg"))
            if not img_paths:
                raise ValueError(f"No images found in {folder}!")
    
            # 随机抽取最多10张图片
            sample_size = min(sample_num, len(img_paths))  # 如果图片不足10张，取全部
            sampled_img_paths = random.sample(img_paths, sample_size)  # 随机采样
            all_mean_acts = []
            for img_path in sampled_img_paths:
                mean_act = self.compute_single_layer_mean_activation(
                    img_path, target_layer_path, left_filter_idx, right_filter_idx, size
                )
                all_mean_acts.append(mean_act)
            return torch.stack(all_mean_acts).cpu().numpy() if isinstance(all_mean_acts[0], torch.Tensor) else np.array(all_mean_acts)
        
        def plot_multi_img_single_layer_mean_activations():
            # 创建双子图布局
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), 
                                        gridspec_kw={'height_ratios': [2, 1]})  # 主图占2份高度，差值图占1份
            
            # 定义颜色和线型方案
            colors = ['b', 'r']
            linestyle_dict = {
                self.origin_img_dir: '--', 
                self.attack_img_dir: '-'
            }
            
            # 绘制原始数据（组间比较）
            legend_handles = []  # 存储每组的句柄（仅两个）
            for group_idx, (folder, data, linestyle) in enumerate([
                (self.origin_img_dir, group1_data, linestyle_dict[self.origin_img_dir]),
                (self.attack_img_dir, group2_data, linestyle_dict[self.attack_img_dir])
            ]):
                folder_name = os.path.basename(folder)
                num_images = data.shape[0] # type: ignore
                for img_idx in range(num_images):
                    activations = data[img_idx] # type: ignore
                    # 绘制线条，所有图像共享组别标签（folder_name）
                    line = ax1.plot(
                        np.arange(data.shape[1]), # type: ignore
                        activations,
                        linestyle=linestyle,
                        color=colors[group_idx],  # 固定颜色
                        alpha=0.8,
                        linewidth=0.5,
                        label=folder_name  # 所有图像使用相同标签
                    )[0]
                    # 每组仅保存第一个图像的句柄用于图例
                    if img_idx == 0:
                        legend_handles.append(line)
                        
            # 设置主图属性（图例仅显示两组）
            ax1.set_title(f"Mean Activation Comparison ({target_layer_path})")
            ax1.set_ylabel("Activation Magnitude")
            ax1.grid(linestyle='--', alpha=0.5)
            ax1.legend(
                handles=legend_handles,
                labels=['origin', 'attack'],
                loc="upper right",
                ncol=1,
                fontsize=8
            )
            # 绘制差值图
            ax2.plot(
                np.arange(len(self.diff)),
                self.diff,
                color='g',
                linestyle='-',
                linewidth=2,
                label='Difference (Group1 - Group2)'
            )
            # ax2.axhline(-1 * activate_threshold, color='dimgrey', linestyle='--', linewidth=1.2)  # 添加阈值参考线
            # ax2.axhline(activate_threshold, color='dimgrey', linestyle='--', linewidth=1.2)  # 添加阈值参考线
            if len(exceeding_indices) > 0:
                significant_x = np.array(exceeding_indices)
                significant_y = self.diff[significant_x]
                
                ax2.scatter(
                    significant_x,
                    significant_y,
                    color='r',
                    s=20,  # 点大小
                    zorder=5,
                    edgecolor='k',
                    linewidth=1,
                    label=f'Significant Differences (n={len(exceeding_indices)})'
                )
                
                # 添加零线参考线（保持原有效果）
                ax2.axhline(0, color='dimgrey', linestyle='--', linewidth=1.2)
                
            # 图例更新
            legend_elements = [
                plt.Line2D([0], [0], color='g', lw=2, label='Difference Curve'),
                plt.Line2D([0], [0], marker='o', color='w', label=f'Significant Points (n={len(exceeding_indices)})',
                        markerfacecolor='r', markersize=10, markeredgecolor='k')
            ]
            ax2.legend(handles=legend_elements, loc="upper right", fontsize=8)
            
            plt.tight_layout()
            plt.show()
        
        outlier_analyzer = OutlierAnalyzer()
            
        group1_data = process_folder(50, self.origin_img_dir)
        group2_data = process_folder(50, self.attack_img_dir)
        
        if group1_data.shape[1] != group2_data.shape[1]:
            raise ValueError("Both groups must have the same number of filters for difference calculation.")
        
        group1_avg = group1_data.mean(axis=0)
        group2_avg = group2_data.mean(axis=0)
        self.diff = group1_avg - group2_avg
        exceeding_indices = outlier_analyzer.threshold_dbscan(self.diff)
        if show: plot_multi_img_single_layer_mean_activations()
        if not show: print(f"{target_layer_path}: index of outlier value: {exceeding_indices}")
        return exceeding_indices
        
    def analyze_layers(self, layer_configs, show=False):
        results = {}
        for config in layer_configs:
            target_layer_path, left, right = config
            result = self.multi_img_single_layer_mean_activations(
                target_layer_path=target_layer_path,
                left_filter_idx=left,
                right_filter_idx=right,
                show=show
            )
            results[target_layer_path] = result
        self.activate_results = results
        return results
    

class LayerConfig(Enum):
    # 格式定义：(索引, [(层路径, 左边界, 右边界), ...])
    backbone_c2f1 = (
        2,
        [
            ('backbone_c2f1.cv1', 0, 32),
            ('backbone_c2f1.cv2', 0, 32),
            ('backbone_c2f1.m.0.cv1', 0, 16),
            ('backbone_c2f1.m.0.cv2', 0, 16),
        ]
    )
    backbone_c2f2 = (
        4,
        [
            ('backbone_c2f2.cv1', 0, 64),
            ('backbone_c2f2.cv2', 0, 64),
            ('backbone_c2f2.m.0.cv1', 0, 32),
            ('backbone_c2f2.m.0.cv2', 0, 32),
            ('backbone_c2f2.m.1.cv1', 0, 32),
            ('backbone_c2f2.m.1.cv2', 0, 32),
        ]
    )
    backbone_c2f3 = (
        6,
        [
            ('backbone_c2f3.cv1', 0, 128),
            ('backbone_c2f3.cv2', 0, 128),
            ('backbone_c2f3.m.0.cv1', 0, 64),
            ('backbone_c2f3.m.0.cv2.conv', 0, 64),
            ('backbone_c2f3.m.1.cv1', 0, 64),
            ('backbone_c2f3.m.1.cv2', 0, 64),
        ]
    )
    backbone_c2f4 = (
        8,
        [
            ('backbone_c2f4.cv1', 0, 256),
            ('backbone_c2f4.cv2', 0, 256),
            ('backbone_c2f4.m.0.cv1', 0, 128),
            ('backbone_c2f4.m.0.cv2', 0, 128),
        ]
    )
    backbone_sppf = (
        9,
        []
    )
    neck_c2f1 = (
        12,
        [
            ('neck_c2f1.cv1', 0, 128),
            ('neck_c2f1.cv2', 0, 128),
            ('neck_c2f1.m.0.cv1', 0, 64),
            ('neck_c2f1.m.0.cv2', 0, 64),
        ]
    )
    neck_c2f2 = (
        15,
        [
            ('neck_c2f2.cv1', 0, 64),
            ('neck_c2f2.cv2', 0, 64),
            ('neck_c2f2.m.0.cv1', 0, 32),
            ('neck_c2f2.m.0.cv2', 0, 32),
        ]
    )
    neck_c2f3 = (
        18,
        [
            ('neck_c2f3.cv1', 0, 128),
            ('neck_c2f3.cv2', 0, 128),
            ('neck_c2f3.m.0.cv1', 0, 64),
            ('neck_c2f3.m.0.cv2', 0, 64),
        ]
    )
    neck_c2f4 = (
        21,
        [
            ('neck_c2f4.cv1', 0, 256),
            ('neck_c2f4.cv2', 0, 256),
            ('neck_c2f4.m.0.cv1', 0, 128),
            ('neck_c2f4.m.0.cv2', 0, 128),
        ]
    )
    def __init__(self, index, layers):
        self.index = index
        self.layers = layers


def analysis_all_layer(model, origin_img_dir, attack_img_dir, show):
    print('Analyzing the index of conv layer where the diff exceeds a certain threshold...')
    result_all_layer = []
    for config in LayerConfig:
        module_index = config.index
        layer_configs = config.layers
        parent_path = config.name # 如 BACKBONE_C2F1 → backbone.c2f1
        # 获取目标模块
        try:
            module = model.model.model[module_index]
        except IndexError:
            print(f"[WARNING] Module index {module_index} out of range, skipping '{parent_path}'")
            continue
        # 创建分析器并执行分析
        analyzer = FilterAnalyzer(
            model=model,
            module=module,
            parent_path=parent_path,
            origin_img_dir=origin_img_dir,
            attack_img_dir=attack_img_dir
        )
        result_all_layer.append(analyzer.analyze_layers(layer_configs, show=show))
    result_all_layer_merged_dict = {k: v for d in result_all_layer for k, v in d.items()}
    return result_all_layer_merged_dict

def analysis_all_layer_cross_classes(yolo, sample_classes=['0', '2', '14', '23', '39']):
    # 存储每个类别的分析结果（包含索引和激活值）
    class_results = {}
    for sample_class in sample_classes:
        attack_img_dir = f'../../gtsrb_classes_attacked_test/masked_pgd_0-0500_0-0005_100/{sample_class}/images/'
        origin_img_dir = f'../../gtsrb_classes_attacked_test/origin/{sample_class}/images/'
        result = analysis_all_layer(yolo, origin_img_dir, attack_img_dir, show=False)
        class_results[sample_class] = result

    reference_layer_names = list(class_results[sample_classes[0]].keys())
    cross_layer_results = {}

    for layer_name in reference_layer_names:
        indices_list = [res[layer_name]['exceeding_indices'] for res in class_results.values() if layer_name in res]
        if not indices_list:
            continue
        # 计算交集
        common_indices = set.intersection(*map(set, indices_list))
        # 如果交集为空，保留空数组
        if not common_indices:
            cross_layer_results[layer_name] = {
                'exceeding_indices': [],
                'origin_avg': np.array([], dtype=np.float32),
                'attack_avg': np.array([], dtype=np.float32)
            }
            continue
        
        reference_indices = class_results[sample_classes[0]][layer_name]['exceeding_indices']
        sorted_indices = [idx for idx in reference_indices if idx in common_indices]
        origin_avgs = []
        attack_avgs = []
        for res in class_results.values():
            if layer_name in res:
                group1_avg = res[layer_name]['origin_avg'][sorted_indices]
                group2_avg = res[layer_name]['attack_avg'][sorted_indices]
                origin_avgs.append(group1_avg)
                attack_avgs.append(group2_avg)
                
        cross_layer_results[layer_name] = {
            'exceeding_indices': sorted_indices,
            'origin_avg': np.array(origin_avgs, dtype=np.float32),
            'attack_avg': np.array(attack_avgs, dtype=np.float32)
        }

    print('''The following are the intersections of critical convolution kernel indices across categories.\n'''
          '''These kernels are significantly affected in attacks across all categories.\n''')
    for layer_name, key_idx in cross_layer_results.items():
        print(f'{layer_name}: {key_idx["exceeding_indices"]}')

    return cross_layer_results



class OutlierAnalyzer:
    def __init__(self):
        pass

    def threshold_static(self, diff, per):
        abs_diff = np.abs(diff)
        threshold = abs_diff.max() * per
        return np.where(abs_diff > threshold)[0].tolist()

    def threshold_iqr(self, diff):
        abs_diff = np.abs(diff)
        Q1 = np.percentile(abs_diff, 25)
        Q3 = np.percentile(abs_diff, 75)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5 * IQR
        return np.where(abs_diff > threshold)[0].tolist()
    
    def threshold_skewness(self, diff):
        abs_diff = np.abs(diff)
        skewness = skew(abs_diff)
        dynamic_ratio = 0.75 * np.exp(-0.1 * np.abs(skewness))
        threshold = np.max(abs_diff) * dynamic_ratio
        return np.where(abs_diff > threshold)[0].tolist()
    
    def threshold_dbscan(self, diff):
        # 数据标准化
        scaler = StandardScaler()
        diff_scaled = scaler.fit_transform(diff.reshape(-1, 1)).flatten()
        # 自动参数优化
        eps_opt = self._auto_select_eps(diff_scaled)
        min_samples_opt = self._auto_select_min_samples(len(diff_scaled))
        # DBSCAN建模
        db = DBSCAN(eps=eps_opt, min_samples=min_samples_opt).fit(diff_scaled.reshape(-1, 1))
        noise_mask = (db.labels_ == -1)
        return np.where(noise_mask)[0].tolist()
    
    def _auto_select_eps(self, data_scaled, k=None):
        n_samples = len(data_scaled)
        k = k or max(2, int(np.log(n_samples)))  # 默认k=log(n)
        
        # 计算k-距离
        nbrs = NearestNeighbors(n_neighbors=k).fit(data_scaled.reshape(-1, 1))
        distances, _ = nbrs.kneighbors(data_scaled.reshape(-1, 1))
        k_distances = np.sort(distances[:, -1])[::-1]  # 降序排列
        
        # 自动检测肘点
        kl = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='decreasing')
        knee_idx = kl.knee or int(n_samples * 0.1)  # 默认前10%样本位置
        return k_distances[knee_idx] if knee_idx < len(k_distances) else k_distances[-1]
    
    def _auto_select_min_samples(self, n_samples):
        return max(2, int(np.log(n_samples)))