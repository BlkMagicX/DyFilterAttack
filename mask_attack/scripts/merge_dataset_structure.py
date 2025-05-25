import os
import shutil
from tqdm import tqdm

def merge_dataset_structure(
    source_base: str = "E://bmx/YoloCAM/mask_attack/result",
    target_base: str = "E://bmx/YoloCAM/mask_attack/no_classes_result"
):
    """
    自动合并攻击数据集（自动检测攻击方法目录）
    
    源结构:
        {source_base}/攻击方法/类别/images/
        {source_base}/攻击方法/类别/labels/
        
    目标结构:
        {target_base}/攻击方法/images/
        {target_base}/攻击方法/labels/
    """
    # 验证源目录存在
    if not os.path.exists(source_base):
        raise FileNotFoundError(f"源目录不存在: {source_base}")
    
    # 自动发现所有攻击方法目录
    methods = [d for d in os.listdir(source_base) 
              if os.path.isdir(os.path.join(source_base, d))]
    
    if not methods:
        print(f"在 {source_base} 中未找到任何攻击方法目录")
        return
    
    print(f"发现 {len(methods)} 个攻击方法: {', '.join(methods)}")

    # 处理每个攻击方法
    for method in tqdm(methods, desc="处理攻击方法", total=len(methods)):
        # 定义路径
        source_method_dir = os.path.join(source_base, method)
        target_method_dir = os.path.join(target_base, method)
        
        # 创建目标目录
        os.makedirs(os.path.join(target_method_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_method_dir, "labels"), exist_ok=True)
        
        # 获取所有类别目录
        try:
            class_dirs = [d for d in os.listdir(source_method_dir) 
                         if os.path.isdir(os.path.join(source_method_dir, d))]
        except Exception as e:
            print(f"无法读取 {method} 的类别目录: {e}")
            continue
            
        # 合并每个类别
        for class_dir in tqdm(class_dirs, desc=f"合并 {method} 类别", 
                            total=len(class_dirs), leave=False):
            src_images = os.path.join(source_method_dir, class_dir, "images")
            src_labels = os.path.join(source_method_dir, class_dir, "labels")
            
            # 验证必要目录存在
            if not (os.path.exists(src_images) and os.path.exists(src_labels)):
                tqdm.write(f"跳过无效目录 {class_dir}: 缺少 images 或 labels 子目录")
                continue
                
            # 复制图像文件
            for img_file in os.listdir(src_images):
                src_path = os.path.join(src_images, img_file)
                dst_path = os.path.join(target_method_dir, "images", img_file)
                if not os.path.exists(dst_path):
                    try:
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        tqdm.write(f"复制图像文件失败 {img_file}: {e}")
            
            # 复制标签文件
            for lbl_file in os.listdir(src_labels):
                src_path = os.path.join(src_labels, lbl_file)
                dst_path = os.path.join(target_method_dir, "labels", lbl_file)
                if not os.path.exists(dst_path):
                    try:
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        tqdm.write(f"复制标签文件失败 {lbl_file}: {e}")

if __name__ == "__main__":
    merge_dataset_structure()