import torch
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import Button, HBox, VBox, Output
from IPython.display import display, clear_output

class CustomDataset(Dataset):
    def __init__(self, images_dir_path, labels_dir_path, image_width, image_height):
        """
        self.image_paths is a list containing the paths of all image files.
        self.label_paths is a list containing the paths of all label files,
        with each label file path corresponding one-to-one with the respective image file path.
        """
        super().__init__()
        self.image_paths = sorted(Path(images_dir_path).glob('*.png'), key=lambda x: x.stem)
        self.label_paths = [Path(labels_dir_path)/f'{p.stem}.txt' for p in self.image_paths]
        self.origin_images_dir_path = images_dir_path 
        self.origin_labels_dir_path = labels_dir_path

        self.image_width = image_width
        self.image_height = image_height
        
    def __len__(self):
        return len(self.image_paths)

    def load_image(self, img_path):
        img_path = str(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(self.image_width, self.image_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # Adjust the dimension order (from (height, width, channels) to (channels, height, width))
        # and normalize the pixel values to the range of [0, 1].
        img = torch.tensor(img).permute(2,0,1) / 255.0  # Normalize and permute dimensions
        return img

    def load_label(self, labels_path):
        labels = torch.tensor([list(map(float, line.split())) for line in open(labels_path)])
        return labels

    def yolo_to_pixel(self, yolo_bboxes):
        """Convert YOLO format [xc,yc,w,h] to pixel coordinates [xmin,ymin,xmax,ymax]"""
        xc = yolo_bboxes[:, 0] * self.image_width
        yc = yolo_bboxes[:, 1] * self.image_height
        w = yolo_bboxes[:, 2] * self.image_width
        h = yolo_bboxes[:, 3] * self.image_height

        x_min = xc - w/2
        y_min = yc - h/2
        x_max = xc + w/2
        y_max = yc + h/2

        bbox_np = torch.from_numpy(np.stack([x_min, y_min, x_max, y_max], axis=0))
        
        return bbox_np

    def convert_yolo_to_batch_format_torch(self, labels, idx):
        """
        Convert YOLO format labels to batch-compatible format for detection tasks

        Args:
            idx (int): Sample index in dataset

        Returns:
            tuple: (batch_indices, classes, bboxes) as torch.Tensor or None if empty
        """
        if labels.numel() == 0:
            return (
            torch.empty(0, 1, dtype=torch.long),
            torch.empty(0, 1, dtype=torch.long),
            torch.empty(0, 4, dtype=torch.float32)
        )
        
        classes = torch.Tensor(labels[:, 0].long()).unsqueeze(1)
        bboxes = self.yolo_to_pixel(labels[:, 1:5]).view(1, -1)
        batch_indices = torch.full((classes.size(0),), fill_value=idx, dtype=torch.int64).unsqueeze(1)
        return batch_indices, classes, bboxes

    def generate_mask(self, bboxes):
        """
        Generate binary mask for object regions from bounding bboxes

        Args:
            bboxes (torch.Tensor): Bounding bboxes in [xmin, ymin, xmax, ymax] format,
                                shape: (N, 4) where N is number of bboxes

        Returns:
            torch.Tensor: Binary mask of shape (image_height, image_width)
                        where 1 indicates object regions
        """
        # Initialize empty mask
        mask = torch.zeros((self.image_height, self.image_width),
                          dtype=torch.float32,
                          device=bboxes.device if torch.is_tensor(bboxes) else None)

        # Early return if no bboxes
        if bboxes.numel() == 0:
            return mask.unsqueeze(0).expand(3, -1, -1)
        # Convert coordinates to integers and clamp to image boundaries
        """Bounding Boxes example:
        [[378.2500, 384.5000, 398.7500, 415.5000],
        [220.5000, 424.5000, 226.5000, 435.5000],
        [227.0000, 424.7500, 233.0000, 436.2500]]
        """
        tmp_bboxes = bboxes.view(-1, 4).int()  # Avoid modifying original tensor
        tmp_bboxes[:, [0, 2]] = tmp_bboxes[:, [0, 2]].clamp(0, self.image_width - 1)
        tmp_bboxes[:, [1, 3]] = tmp_bboxes[:, [1, 3]].clamp(0, self.image_height - 1)

        # Vectorized implementation (faster than loop for multiple bboxes)
        for xmin, ymin, xmax, ymax in tmp_bboxes:
            mask[ymin: ymax + 1, xmin: xmax + 1] = 1

        return mask.unsqueeze(0).expand(3, -1, -1)

    def __getitem__(self, idx):
        img = self.load_image(self.image_paths[idx])
        label = self.load_label(self.label_paths[idx])
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        batch_indices, classes, bboxes = self.convert_yolo_to_batch_format_torch(label, idx)
        mask = self.generate_mask(bboxes)
        dict = {
          "image": img,
          "label": label,
          "image_path": image_path,
          "label_path": label_path,
          "batch_indices": batch_indices,
          "classes": classes,
          "bboxes": bboxes,
          "mask": mask
        }
        # print("path: ", self.label_paths[idx])
        return dict
    
    
def check_image_sizes(folder_path):
    """
    Check the dimensions of all .jpg images in a folder and return a list of all found dimensions.

    Args:
        folder_path (str): Path to the image folder

    Returns:
        tuple: (whether all images have the same dimensions, list of all found dimensions)
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"{folder_path} is not a valid folder path")
    sizes = set()  # Use a set to automatically remove duplicates
    inconsistent = False
    # Iterate through all .jpg files in the folder
    for img_path in folder.glob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                sizes.add(img.size)  # img.size returns (width, height)
        except Exception as e:
            print(f"Unable to read image {img_path}: {e}")
            continue
    # Check if all images have the same dimensions
    all_same = len(sizes) == 1
    return all_same, sorted(sizes)  # Return a sorted list of dimensions


def custom_collate_fn(batch):
    """
    Custom collate function to handle batch processing with padding.
    
    Filters out samples with empty labels and returns a batch dictionary.
    
    Args:
        batch (list): List of dictionaries containing image, mask, labels, etc.
        
    Returns:
        dict: Batched data with padded sequences, or None if all samples are filtered out.
    """
    # Filter out samples with empty labels
    batch = [sample for sample in batch if len(sample["label"]) > 0]
    if len(batch) == 0:
        return None  # Return None if no valid samples remain

    # Collect data from each sample
 
    image_path = [sample['image_path'] for sample in batch]        # List[(Path)]
    label_path = [sample['label_path'] for sample in batch]       # List[(Path)]
    label = [sample['label'] for sample in batch]                         # List[(N_i, D)]
    
    image = torch.stack([sample['image'] for sample in batch], dim=0)     # (B, C, H, W)
    mask = torch.stack([sample['mask'] for sample in batch], dim=0)       # (B, C, H, W)
    batch_indices = torch.stack([sample['batch_indices'] for sample in batch], dim=0)
    classes = torch.stack([sample['classes'] for sample in batch], dim=0)
    bboxes = torch.stack([sample['bboxes'] for sample in batch], dim=0)
    
    # 4. 执行填充操作
    label = pad_sequence(label, batch_first=True, padding_value=-1)                   # (B, N_max, D)
    batch_indices = pad_sequence(batch_indices, batch_first=True, padding_value=-1)     # (B, N_max, 1)
    classes = pad_sequence(classes, batch_first=True, padding_value=-1)                 # (B, N_max, 1)
    bboxes = pad_sequence(bboxes, batch_first=True, padding_value=-1).squeeze(1)                   # (B, N_max, 4)
    
    # Return batch dictionary
    return {
        "image_path": image_path,
        "label_path": label_path,
        "image": image,
        "mask": mask,
        "label": label,
        "batch_indices": batch_indices,
        "classes": classes,
        "bboxes": bboxes
    }
    
    
# 绘图相关代码，不用动
def draw_boxes(img, predictions, ax, title=""):
    img = img.detach().cpu().numpy()  # Convert tensor to numpy array
    img = np.transpose(img, (1, 2, 0))  # Change the order of dimensions
    img = np.clip(img, 0, 1) * 255  # Convert image back to pixel range [0, 255]
    img = img.astype(np.uint8)  # Convert to uint8 for cv2
    
    print(f'{title}: pred result:')
    
    for pred in predictions:
        box = pred['box']
        score = pred['score']
        label = pred['label']

        # Draw bounding box
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

        # Modify text parameters here:
        text = f'{label} ({score:.2f})'
        print(text)
        font_scale = 0.8  # 原值是0.5，调大数值可增大字体（例如改为0.8或1.0）
        thickness = 2     # 字体边框粗细，可与font_scale配合调整

        # 使用更清晰的字体（可选）
        font = cv2.FONT_HERSHEY_DUPLEX  # 原默认是cv2.FONT_HERSHEY_SIMPLEX

        img = cv2.putText(
            img,
            text,
            (int(box[0]), int(box[1]) - 10),
            font,           # 使用新字体
            font_scale,     # 新字体大小
            (255, 0, 0),   # 文字颜色
            thickness       # 字体粗细
        )

    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')    
    
    
# Function to visualize a pair of images
def visualize_image_pair(model, index, origin_dataset, attacked_dataset):
    clear_output() 
    # Get the original and adversarial images
    img = origin_dataset[index]["image"]
    perturbed_img = attacked_dataset[index]["image"]

    # Get predictions for both images
    original_preds = model(img.unsqueeze(0))
    adversarial_preds = model(perturbed_img.unsqueeze(0))

    # Extract useful prediction data (bounding boxes, scores, and labels)
    original_results = [
        {'box': pred[:4], 'score': pred[4], 'label': model.names[int(pred[5])]}
        for pred in original_preds[0].boxes.data
    ]
    adversarial_results = [
        {'box': pred[:4], 'score': pred[4], 'label': model.names[int(pred[5])]}
        for pred in adversarial_preds[0].boxes.data
    ]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(30, 14))

    # Draw the original image with predictions
    draw_boxes(img.squeeze(), original_results, axs[0], title="Original Image Predictions")

    # Draw the adversarial image with predictions
    draw_boxes(perturbed_img.squeeze(), adversarial_results, axs[1], title="Adversarial Image Predictions")

    # Show the plot
    plt.show()
    
    
def visualize_attack_result(model, origin_dataset, attacked_dataset):
    dataset_size = len(origin_dataset)
    current_index = 5

    prev_button = Button(description="Previous")
    next_button = Button(description="Next")
    index_label = Button(description=f"Current Index: {current_index}", disabled=True)
    output = Output()

    def update_display(index):
        nonlocal current_index
        current_index = index
        index_label.description = f"Current Index: {current_index}"
        with output:
            clear_output()
            visualize_image_pair(model, current_index, origin_dataset, attacked_dataset)

    def on_prev_click(b):
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            update_display(current_index)

    def on_next_click(b):
        nonlocal current_index
        if current_index < dataset_size - 1:
            current_index += 1
            update_display(current_index)

    prev_button.on_click(on_prev_click)
    next_button.on_click(on_next_click)

    update_display(0)

    controls = HBox([prev_button, next_button, index_label])
    display(VBox([controls, output]))
