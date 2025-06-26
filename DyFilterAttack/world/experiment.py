from dataclasses import dataclass, asdict
from typing import Dict, List, Callable
import torch

# 用字符串占位，真正实例化 Optimizer 时再传 params 和 lr
OPTIMIZER_FACTORY: Dict[str, Callable] = {
    "adam": lambda p, lr: torch.optim.Adam(p, lr=lr),
    "sgd": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
    "adamw": lambda p, lr: torch.optim.AdamW(p, lr=lr),
}


@dataclass
class ExpConfig:
    name: str
    hyp: Dict  # e.g. {"eps": 8/255, "lr": 1e-2, "num_iter": 40}
    target_layer: str
    optim_name: str  # "adam" / "sgd" / ...

    def build_optimizer(self, delta):
        return OPTIMIZER_FACTORY[self.optim_name.lower()]([delta], lr=self.hyp["lr"])
