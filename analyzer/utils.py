import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from functools import partial
import torchvision.transforms as T
import os
import glob

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
            print(f"    Registering Hook: {current_path}")
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
        self.model, self.module, self.parent_path = model, module, parent_path
        self.origin_img_dir, self.attack_img_dir = origin_img_dir, attack_img_dir
        self.target_layer_path = None
        self.model.model.eval()
        self.save_features = SaveFeatures()
        print(f"For Parent Path: {self.parent_path}")
        self.save_features.register_hooks(module=self.module, parent_path=self.parent_path)
        
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
        img_var = torch.tensor(data=img_tensor, requires_grad=True, device=self.model.device, dtype=torch.float32)
        
        with torch.no_grad():
            _ = self.model.model(img_var)  # YOLOv8 前向传播（输出可能包含多个特征图，但钩子仅捕获目标层）
            activations = self.save_features.get_features()
            target_features = activations[target_layer_path][0, left_filter_idx:right_filter_idx, :, :] 
            mean_activation = np.float32(target_features.mean(axis=[1, 2]).cpu().detach().numpy())
        
        # # 绘制单图
        # filter_num = right_filter_idx - left_filter_idx
        # plt.bar(range(filter_num), mean_activation, color='blue', alpha=1)
        # plt.title(f'Mean Activation per Filter: {target_layer_path}')
        # plt.xlabel('Filter Index')
        # plt.ylabel('Activation Magnitude')
        # plt.xticks(np.arange(0, filter_num, max(1, filter_num//10)))  # 自动调整刻度密度
        # plt.grid(axis='y', linestyle='--', alpha=0.5)
        # plt.tight_layout()
        # plt.show()
        
        mean_activation = torch.from_numpy(mean_activation).to(device=self.model.device, dtype=torch.float32)
        
        return mean_activation

    def multi_img_single_layer_mean_activations(self, target_layer_path, left_filter_idx, right_filter_idx, size=640, show=True):
        def process_folder(folder):
            img_paths = glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg"))
            if not img_paths:
                raise ValueError(f"No images found in {folder}!")
            all_mean_acts = []
            for img_path in img_paths:
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
            ax2.axhline(0, color='dimgrey', linestyle='--', linewidth=1.2)  # 添加零线参考线
            ax2.axhline(-1 * activate_threshold, color='dimgrey', linestyle='--', linewidth=1.2)  # 添加阈值参考线
            ax2.axhline(activate_threshold, color='dimgrey', linestyle='--', linewidth=1.2)  # 添加阈值参考线
            ax2.set_title("Average Activation Difference")
            ax2.set_xlabel("Filter Index")
            ax2.set_ylabel("Difference (Group1 - Group2)")
            ax2.grid(linestyle='--', alpha=0.5)
            ax2.legend(loc="upper right", fontsize=8)
            
            plt.tight_layout()
            plt.show()
        
            # 返回两个组的张量数据（保持原有返回值）
            return exceeding_indices
        
        group1_data = process_folder(self.origin_img_dir)
        group2_data = process_folder(self.attack_img_dir)
        
        if group1_data.shape[1] != group2_data.shape[1]:
            raise ValueError("Both groups must have the same number of filters for difference calculation.")
        
        group1_avg = group1_data.mean(axis=0)
        group2_avg = group2_data.mean(axis=0)
        self.diff = group1_avg - group2_avg
        activate_threshold = np.abs(self.diff).max() * 0.75
        exceeding_indices = np.where(np.abs(self.diff) > activate_threshold)[0].tolist()  # 转换为Python列表便于后续处理
        print(f"{target_layer_path}: exceed thd {activate_threshold:.2f}: {exceeding_indices}")
        if show: plot_multi_img_single_layer_mean_activations()
    
