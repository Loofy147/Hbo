"""Robust genome -> PyTorch model builder. Improved: supports config-driven modules and optional dropouts."""
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn

@dataclass
class Genome:
    architecture: List[str]
    lr: float = 1e-3
    wd: float = 5e-4
    batch_size: int = 128
    meta: Optional[dict] = None

class SimpleNet(nn.Module):
    def __init__(self, layers: nn.Module):
        super().__init__()
        self.net = layers

    def forward(self, x):
        return self.net(x)

def _conv_layer(cin, cout, k=3, stride=1, padding=None):
    if padding is None:
        padding = k // 2
    return nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=padding)

def build_model_from_genome(genome: Genome, in_channels=3, num_classes=10, use_dropout=False) -> nn.Module:
    modules = []
    cur_c = in_channels
    is_flat = False
    used_lazy = False

    for token in genome.architecture:
        t = token.lower().strip()
        if is_flat and (t.startswith('conv') or t.startswith('pool') or t == 'bn'):
            continue

        if t.startswith('conv'):
            parts = t.split('-')
            k = 3
            if '5' in parts[0]:
                k = 5
            out_ch = int(parts[1]) if len(parts) > 1 else cur_c
            modules.append(_conv_layer(cur_c, out_ch, k=k))
            modules.append(nn.ReLU(inplace=True))
            cur_c = out_ch
            is_flat = False

        elif t.startswith('sep_conv'):
            parts = t.split('-')
            k = 3
            out_ch = int(parts[1]) if len(parts) > 1 else cur_c
            modules.append(nn.Conv2d(cur_c, cur_c, kernel_size=k, padding=k//2, groups=cur_c))
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(cur_c, out_ch, kernel_size=1))
            modules.append(nn.ReLU(inplace=True))
            cur_c = out_ch
            is_flat = False

        elif t == 'pool-max':
            modules.append(nn.MaxPool2d(2))
            is_flat = False

        elif t == 'pool-avg':
            modules.append(nn.AvgPool2d(2))
            is_flat = False

        elif t == 'global_pool_avg':
            modules.append(nn.AdaptiveAvgPool2d((1, 1)))
            modules.append(nn.Flatten())
            is_flat = True

        elif t == 'relu':
            modules.append(nn.ReLU(inplace=True))

        elif t == 'bn':
            if not is_flat:
                modules.append(nn.BatchNorm2d(cur_c))
            else:
                modules.append(nn.BatchNorm1d(cur_c if isinstance(cur_c, int) else 1))

        elif t == 'flatten':
            modules.append(nn.Flatten())
            is_flat = True

        elif t.startswith('fc'):
            parts = t.split('-')
            out_dim = int(parts[1]) if len(parts) > 1 else 128
            if not is_flat:
                modules.append(nn.AdaptiveAvgPool2d((1, 1)))
                modules.append(nn.Flatten())
                is_flat = True
            modules.append(nn.LazyLinear(out_dim))
            modules.append(nn.ReLU(inplace=True))
            if use_dropout:
                modules.append(nn.Dropout(0.2))
            used_lazy = True
            cur_c = out_dim

        else:
            # ignore unknown tokens but log could be added
            continue

    if not is_flat:
        modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        modules.append(nn.Flatten())
        modules.append(nn.Linear(cur_c, num_classes))
    else:
        last_mod = modules[-1] if modules else None
        if not isinstance(last_mod, nn.Linear) and not isinstance(last_mod, nn.LazyLinear):
            modules.append(nn.Linear(cur_c, num_classes))

    model = SimpleNet(nn.Sequential(*modules))
    model._meta = {'used_lazy_linear': used_lazy}
    return model