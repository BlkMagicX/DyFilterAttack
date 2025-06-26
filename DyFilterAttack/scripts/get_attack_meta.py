#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_attack_meta.py
~~~~~~~~~~~~~~~~~~
🎯 生成针对 YOLOWorld 的攻击元数据 attack_meta.json
   - 数据源: LVIS-v1 val (标注) + COCO2017 val 图像
   - 过滤规则:
       1) YOLOWorld 在 conf > conf_thres 时检出 >=1 个框
       2) 该框的 Top-1 与 Top-2 类别不同，且 logit_1 - logit_2 > margin_pre
"""

import argparse, json, os, shutil, tarfile, zipfile, tempfile, urllib.request
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request, shutil, time, json

import numpy as np
import torch
from ultralytics.utils import ops
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLOWorld


# --------------------------------------------------------------------------- #
#                               下载工具                                       #
# --------------------------------------------------------------------------- #
_COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
_LVIS_VAL_JSON = "https://storage.googleapis.com/sfr-vision-language-research/LVIS/lvis_v1_val.json"
# --- 备用镜像 URL ---
_LVIS_VAL_JSON_MIRRORS = [
    # HuggingFace
    "https://huggingface.co/datasets/visual_genome/LVIS/resolve/main/lvis_v1_val.json",
    # GitHub raw
    "https://raw.githubusercontent.com/zhangxiaosong18/LVIS-dataset/master/lvis_v1_val.json",
]


def download(url: str, save_path: Path, desc: str = "", mirrors: list[str] | None = None) -> None:
    """带 UA + 镜像重试的下载器"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        print(f"[✓] {desc or save_path.name} 已存在，跳过下载")
        return

    urls = [url] + (mirrors or [])
    for idx, u in enumerate(urls, 1):
        try:
            print(f"↓ ({idx}/{len(urls)}) 正在下载 {desc or u}")
            req = urllib.request.Request(
                u,
                headers={"User-Agent": "Mozilla/5.0"},  # 添加 UA 绕过 403
            )
            with urllib.request.urlopen(req) as resp, open(save_path, "wb") as f:
                shutil.copyfileobj(resp, f)
            print(f"[✓] 下载完成 -> {save_path}")
            return
        except urllib.error.HTTPError as e:
            print(f"[×] HTTP {e.code} - {e.reason}，尝试下一个镜像...")
        except Exception as e:
            print(f"[×] 下载失败: {e}，尝试下一个镜像...")
        time.sleep(1)

    raise RuntimeError(f"所有镜像均下载失败，请手动下载 {desc or url} 到 {save_path}")


def extract_zip(zip_path: Path, extract_to: Path):
    print(f"• 正在解压 {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"[✓] 解压完成")


# --------------------------------------------------------------------------- #
#                                元数据生成                                   #
# --------------------------------------------------------------------------- #
def coco_val_images(root: Path) -> List[Path]:
    """返回 COCO val2017 中所有 .jpg 路径"""
    img_dir = root / "val2017"
    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} 不存在，请检查解压")
    return sorted(img_dir.glob("*.jpg"))


def load_image(path: Path, size: int = 640) -> torch.Tensor:
    """加载并 letterbox resize 到 size × size，返回 tensor(B=1,C,H,W) ∈[0,1]"""
    im = Image.open(path).convert("RGB")
    im = im.resize((size, size))
    arr = np.array(im).astype(np.float32) / 255.0  # H,W,C
    arr = arr.transpose(2, 0, 1)  # C,H,W
    return torch.from_numpy(arr).unsqueeze(0)  # 1,C,H,W


@torch.no_grad()
def run_world_on_image(model, img_tensor, conf_thres=0.25):
    """返回 (pred_logits, det_indices)"""
    preds, _ = model.model(img_tensor)  # (B, 4+nc, N)
    model.predictor = model._smart_load("predictor")(_callbacks=model.callbacks)
    model.predictor.setup_model(model=model.model, verbose=False)
    predictor = model.predictor
    predictor = model.predictor
    logits = preds[:, 4:, :]  # (B, nc, N)
    # NMS (Ultralytics util)
    detections, keep_idxs = ops.non_max_suppression(
        preds,
        conf_thres,
        predictor.args.iou,
        predictor.args.classes,
        predictor.args.agnostic_nms,
        predictor.args.max_det,
        nc=0 if predictor.args.task == "detect" else len(predictor.model.names),
        end2end=getattr(predictor.model, "end2end", False),
        rotated=predictor.args.task == "obb",
        return_idxs=True,
    )

    return logits[0], keep_idxs[0]  # (nc,N)  &  idx Tensor(k,)


def build_attack_meta(
    world,
    img_paths: List[Path],
    out_json: Path,
    conf_thres: float = 0.25,
    margin_pre: float = 0.0,
) -> None:
    """
    推理 + 过滤，生成 attack_meta.json
    """
    recs: List[Dict] = []
    pbar = tqdm(img_paths, desc="Scanning LVIS-val")
    for p in pbar:
        img_t = load_image(p).to(world.device)
        logits, keep_idx = run_world_on_image(world, img_t, conf_thres)  # (nc,N)  keep_idx shape(k,)
        if keep_idx.numel() == 0:
            continue

        # 取 Top-2 logits per det
        top2_vals, top2_ids = logits[:, keep_idx].topk(2, dim=0)  # (2, k)
        top1, top2 = top2_vals[0], top2_vals[1]  # shape (k,)
        # 条件: top1 - top2 > margin_pre 且类别不同
        ok_mask = (top1 - top2 > margin_pre) & (top2_ids[0] != top2_ids[1])
        if not ok_mask.any():
            continue

        # 只保存满足条件的 det 索引
        sel = keep_idx[ok_mask.nonzero(as_tuple=True)[0]]
        recs.append({"image": str(p.relative_to(p.parents[1])), "det_ids": sel.tolist()})  # val2017/000000.jpg

        pbar.set_postfix(kept=len(recs))

    print(f"[✓] 最终可攻击图像数: {len(recs)}")
    out_json.write_text(json.dumps(recs, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
#                                   主函数                                    #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser("Generate attack_meta.json for YOLOWorld on LVIS-val")
    parser.add_argument("--save-dir", default="./dataset/LVIS", type=Path, help="数据及 meta 保存根目录")
    parser.add_argument("--model", required=True, type=Path, help="YOLOWorld checkpoint 路径 (*.pt / *.yaml)")
    parser.add_argument("--img-size", default=640, type=int, help="推理分辨率")
    parser.add_argument("--conf", default=0.25, type=float, help="NMS 置信度阈")
    parser.add_argument("--margin-pre", default=0.0, type=float, help="Top1-Top2 logit 差值 > margin 才认为可攻击")
    args = parser.parse_args()

    # ---------------- download & prepare dataset ---------------- #
    coco_zip = args.save_dir / "val2017.zip"
    coco_root = args.save_dir
    lvis_json = args.save_dir / "lvis_v1_val.json"

    download(_COCO_VAL_URL, coco_zip, "COCO val2017 images")
    if not (coco_root / "val2017").exists():
        extract_zip(coco_zip, coco_root)

    download(
        _LVIS_VAL_JSON,
        lvis_json,
        "LVIS val annotation",
        mirrors=_LVIS_VAL_JSON_MIRRORS,
    )

    # ---------------- load model ---------------- #
    world = YOLOWorld(args.model)
    world.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    world.model.eval()
    print(f"[✓] YOLOWorld loaded: {args.model}")

    # ---------------- scan & build meta ---------------- #
    img_paths = coco_val_images(coco_root)
    out_json = args.save_dir / "attack_meta.json"
    build_attack_meta(
        world,
        img_paths,
        out_json,
        conf_thres=args.conf,
        margin_pre=args.margin_pre,
    )
    print(f"[+] attack_meta.json saved to {out_json.resolve()}")


if __name__ == "__main__":
    main()


# python DyFilterAttack/scripts/get_attack_meta.py --model DyFilterAttack/models/yolov8s-world.pt --save-dir DyFilterAttack/testset/lvis_data --conf 0.25 --margin-pre 0.0
