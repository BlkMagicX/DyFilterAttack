import torch
from pathlib import Path
import cv2
import json
import numpy as np
from typing import List, Dict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from ultralytics.data.augment import LetterBox


class CustomDataset(Dataset):
    def __init__(self, images_dir_path, labels_dir_path):
        super().__init__()
        self.image_paths = sorted([p for p in Path(images_dir_path).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}], key=lambda x: x.stem)
        self.label_paths = [Path(labels_dir_path) / f"{p.stem}.txt" for p in self.image_paths]
        self.image_width = 640
        self.image_height = 640
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_paths)

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = image[..., ::-1]
        letterbox = LetterBox(new_shape=(self.image_width, self.image_height), auto=False, scale_fill=False, scaleup=True, stride=32)
        image = letterbox(image=image)
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image)
        img_tensor = torch.from_numpy(image)
        img_tensor = img_tensor.to(self.device)
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
        if labels.numel() == 0:
            return (torch.empty(0, 1, dtype=torch.long), torch.empty(0, 1, dtype=torch.long), torch.empty(0, 4, dtype=torch.float32))

        classes = labels[:, 0].clone().long().unsqueeze(1)  # Shape: (N, 1)
        bboxes = self.yolo_to_pixel(labels[:, 1:5])  # Shape: (N, 4) <-- Corrected, no .view()
        batch_indices = torch.full((classes.size(0),), fill_value=idx, dtype=torch.long).unsqueeze(1)  # Shape: (N, 1)
        return batch_indices, classes, bboxes

    def generate_mask(self, bboxes):
        mask = torch.zeros((self.image_height, self.image_width), dtype=torch.float32, device=bboxes.device if torch.is_tensor(bboxes) else "cpu")

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
            mask[ymin : ymax + 1, xmin : xmax + 1] = 1

        return mask.unsqueeze(0).expand(3, -1, -1)

    def __getitem__(self, idx):
        image = self.preprocess_image(self.image_paths[idx])
        label = self.load_label(self.label_paths[idx])
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        batch_indices, classes, bboxes = self.convert_yolo_to_batch_format_torch(label, idx)
        mask = self.generate_mask(bboxes)
        dict = {
            "image": image,
            "label": label,
            "image_path": image_path,
            "label_path": label_path,
            "batch_indices": batch_indices,
            "classes": classes,
            "bboxes": bboxes,
            "mask": mask,
        }
        return dict


def custom_collate_fn(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = [sample for sample in batch if len(sample["label"]) > 0]
    if not batch:
        return None

    image_path = [sample["image_path"] for sample in batch]
    label_path = [sample["label_path"] for sample in batch]

    image = torch.stack([sample["image"] for sample in batch], dim=0).to(device)
    mask = torch.stack([sample["mask"] for sample in batch], dim=0).to(device)

    labels_list = [sample["label"] for sample in batch]
    batch_indices_list = [sample["batch_indices"] for sample in batch]
    classes_list = [sample["classes"] for sample in batch]
    bboxes_list = [sample["bboxes"] for sample in batch]

    label = pad_sequence(labels_list, batch_first=True, padding_value=-1).to(device)  # (B, N_max, D)
    batch_indices = pad_sequence(batch_indices_list, batch_first=True, padding_value=-1).to(device)  # (B, N_max, 1)
    classes = pad_sequence(classes_list, batch_first=True, padding_value=-1).to(device)  # (B, N_max, 1)
    bboxes = pad_sequence(bboxes_list, batch_first=True, padding_value=-1).to(device)  # (B, N_max, 4)

    return {
        "image_path": image_path,
        "label_path": label_path,
        "image": image,
        "mask": mask,
        "label": label,
        "batch_indices": batch_indices,
        "classes": classes,
        "bboxes": bboxes,
    }


class AttackDataset(Dataset):
    """
    det_ids_batch: List of tensors, one per image in the batch.
      Each tensor contains the indices (in NMS output) of the detection boxes that we want to attack.
      For example, det_ids_batch[b] = Tensor([7, 15, 42]) means:
        - For the b-th image in the batch,
        - Only attack the 7th, 15th, and 42nd detection boxes in the NMS output.

    These indices are selected based on certain criteria (e.g., confidence, class, IoU),
    and are stored in the 'det_ids' field of the attack_meta.json file during preprocessing.

    During training/inference, the DataLoader loads this information and passes it directly
    to the forward() function → compute_gradients_y_det_and_activation().
    """

    IMG_EXT = {".jpg", ".jpeg", ".png"}

    def __init__(self, meta_json: str, images_dir_path: str):
        super().__init__()
        self.records: List[Dict] = json.load(open(meta_json, "r"))
        self.images_dir_path = Path(images_dir_path)
        self.image_width = 640
        self.image_height = 640
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lb = LetterBox(new_shape=(self.image_width, self.image_height), auto=False, scale_fill=False, scaleup=True, stride=32)

    def _read_and_preprocess(self, img_path: Path) -> torch.Tensor:
        im = cv2.imread(str(img_path))
        if im is None:
            raise FileNotFoundError(f"img not found: {img_path}")
        im = im[..., ::-1]  # BGR → RGB
        im = self.lb(image=im)  # letterbox
        im = im.transpose(2, 0, 1)  # C,H,W
        im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
        im_t = torch.from_numpy(im).to(self.device)
        return im_t

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img_rel = rec["image"].replace("\\", "/")
        img_path = self.images_dir_path / img_rel
        img_t = self._read_and_preprocess(img_path)

        det_ids = torch.tensor(rec["det_ids"], dtype=torch.long, device=self.device)
        return {
            "image": img_t,
            "image_path": str(img_path),
            "det_ids": det_ids,
        }


def attack_collate(batch: List[Dict]):
    """
    - images 堆叠 -> Tensor (B,C,H,W)
    - det_ids 保留 list[Tensor]，长度对齐 images
    """
    if batch is None or len(batch) == 0:
        return None

    device = batch[0]["image"].device
    images = torch.stack([b["image"] for b in batch]).to(device)

    image_paths = [b["image_path"] for b in batch]
    det_ids = [b["det_ids"] for b in batch]  # list of Tensor

    return {
        "image": images,
        "image_path": image_paths,
        "det_ids": det_ids,
    }
