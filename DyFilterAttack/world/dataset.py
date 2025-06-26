import torch
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import yaml
import os
from ultralytics.data.augment import LetterBox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, images_dir_path, labels_dir_path, image_width, image_height):
        """
        self.image_paths is a list containing the paths of all image files.
        self.label_paths is a list containing the paths of all label files,
        with each label file path corresponding one-to-one with the respective image file path.
        """
        super().__init__()
        self.image_paths = sorted(
            [p for p in Path(images_dir_path).glob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}],
            key=lambda x: x.stem
        )
        self.label_paths = [Path(labels_dir_path)/f'{p.stem}.txt' for p in self.image_paths]
        self.origin_images_dir_path = images_dir_path 
        self.origin_labels_dir_path = labels_dir_path

        self.image_width = image_width
        self.image_height = image_height
        
    def __len__(self):
        return len(self.image_paths)
    
    def load_image(self, image_path):
        img = cv2.imread(image_path)
        img = img[..., ::-1]
        letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, stride=32)
        img = letterbox(image=img)
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device)
        img_tensor = img_tensor.float()
        img_tensor = img_tensor / 255.0
        return img_tensor

    def load_label(self, labels_path):
        labels = torch.tensor([list(map(float, line.split())) for line in open(labels_path)])
        return labels

    def yolo_to_pixel(self, yolo_bboxes):
        """Convert YOLO format [xc,yc,w,h] to pixel coordinates [xmin,ymin,xmax,ymax]"""
        # yolo_bboxes is a torch.Tensor of shape (N, 4)
        xc = yolo_bboxes[:, 0] * self.image_width
        yc = yolo_bboxes[:, 1] * self.image_height
        w = yolo_bboxes[:, 2] * self.image_width
        h = yolo_bboxes[:, 3] * self.image_height

        x_min = xc - w / 2
        y_min = yc - h / 2
        x_max = xc + w / 2
        y_max = yc + h / 2

        # Stack along axis=1 to create a tensor of shape (N, 4)
        # This avoids converting to numpy and back, and ensures the correct shape.
        return torch.stack([x_min, y_min, x_max, y_max], axis=1)

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
        
        classes = labels[:, 0].clone().long().unsqueeze(1)                   # Shape: (N, 1)
        bboxes = self.yolo_to_pixel(labels[:, 1:5])                        # Shape: (N, 4) <-- Corrected, no .view()
        batch_indices = torch.full((classes.size(0),), fill_value=idx, dtype=torch.long).unsqueeze(1) # Shape: (N, 1)
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
                          device=bboxes.device if torch.is_tensor(bboxes) else 'cpu')

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
    if not batch: # A more Pythonic way to check for an empty list
        return None  # Return None if no valid samples remain

    # Collect data from each sample
    image_path = [sample['image_path'] for sample in batch]
    label_path = [sample['label_path'] for sample in batch]
    
    # For tensors that are already the same size (image and mask), torch.stack is correct.
    image = torch.stack([sample['image'] for sample in batch], dim=0).to(device)
    mask = torch.stack([sample['mask'] for sample in batch], dim=0).to(device)
    
    # For variable-length tensors, collect them into lists first.
    labels_list = [sample['label'] for sample in batch]
    batch_indices_list = [sample['batch_indices'] for sample in batch]
    classes_list = [sample['classes'] for sample in batch]
    bboxes_list = [sample['bboxes'] for sample in batch]
    
    # Now, pad the sequences of these variable-length tensors.
    label = pad_sequence(labels_list, batch_first=True, padding_value=-1).to(device)           # (B, N_max, D)
    batch_indices = pad_sequence(batch_indices_list, batch_first=True, padding_value=-1).to(device) # (B, N_max, 1)
    classes = pad_sequence(classes_list, batch_first=True, padding_value=-1).to(device)       # (B, N_max, 1)
    bboxes = pad_sequence(bboxes_list, batch_first=True, padding_value=-1).to(device)         # (B, N_max, 4)
    
    # Return the correctly batched and padded dictionary
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
    