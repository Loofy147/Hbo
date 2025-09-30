"""evaluate_multifidelity: rewritten to use batched evaluation on torch when possible and vectorized metrics."""
from __future__ import annotations
import math
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from .model_builder import build_model_from_genome, Genome
from .utils import as_f32_contiguous

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model_quick_torch(model: torch.nn.Module, device: torch.device, loader: DataLoader):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            running_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    if total == 0:
        return 0.0, 0.0
    return running_loss / total, correct / total

def estimate_receptive_field(genome: Genome, input_size=32) -> int:
    rf = 1
    stride_acc = 1
    for tok in genome.architecture:
        if tok.startswith('conv') or tok.startswith('sep_conv'):
            parts = tok.split('-')
            k = 3
            if '5' in parts[0]:
                k = 5
            rf = rf + (k - 1) * stride_acc
        elif tok.startswith('pool'):
            stride_acc *= 2
            rf = rf + (2 - 1) * stride_acc
    return int(rf)

def estimate_flops_params(genome: Genome, input_size=32, in_channels=3):
    # same as before (kept), returns flops, params
    flops = 0
    params = 0
    cur_c = in_channels
    h = w = input_size
    for tok in genome.architecture:
        if tok.startswith('conv'):
            parts = tok.split('-')
            k = 3
            if '5' in parts[0]:
                k = 5
            out_ch = int(parts[1]) if len(parts) > 1 else cur_c
            kernel_area = k * k
            fl = kernel_area * cur_c * out_ch * h * w * 2
            pa = cur_c * out_ch * kernel_area
            flops += fl
            params += pa
            cur_c = out_ch
        elif tok.startswith('sep_conv'):
            parts = tok.split('-')
            k = 3
            out_ch = int(parts[1]) if len(parts) > 1 else cur_c
            kernel_area = k * k
            fl = kernel_area * cur_c * h * w * 2 + cur_c * out_ch * h * w * 2
            pa = cur_c * kernel_area + cur_c * out_ch
            flops += fl
            params += pa
            cur_c = out_ch
        elif tok.startswith('pool'):
            h = max(1, h // 2)
            w = max(1, w // 2)
        elif tok.startswith('fc'):
            parts = tok.split('-')
            out_dim = int(parts[1])
            fl = cur_c * out_dim * 2
            pa = cur_c * out_dim
            flops += fl
            params += pa
            cur_c = out_dim
    return flops, params

def composite_fitness_from_metrics(val_acc, params_est, flops, rf, input_size=32,
                                   w_accuracy=1.0, w_params=0.12, w_flops=0.18, rf_bonus_coef=0.25):
    param_pen = 1.0 / (1.0 + math.log1p(params_est))
    flops_pen = 1.0 / (1.0 + math.log1p(flops / 1e6 + 1.0))
    rf_bonus = 1.0 + rf_bonus_coef * min(1.0, rf / (input_size // 2 + 1))
    fitness = (val_acc ** w_accuracy) * (param_pen ** w_params) * (flops_pen ** w_flops) * rf_bonus
    return float(fitness)

def evaluate_multifidelity(genome_dict, data_manager, train_idx, val_idx,
                           device=None, epochs_proxy=2, epochs_full=12,
                           use_full=False, verbose=False, fitness_cfg=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    try:
        genome = Genome(**genome_dict) if not isinstance(genome_dict, Genome) else genome_dict
        X = data_manager.x_full
        Y = data_manager.y_full
        X_arr = np.array(X)
        # convert to CHW
        if X_arr.ndim == 4 and X_arr.shape[-1] in (1, 3):
            X_t = torch.tensor(np.transpose(X_arr, (0, 3, 1, 2)), dtype=torch.float32)
        else:
            X_t = torch.tensor(X_arr, dtype=torch.float32)
        Y_t = torch.tensor(np.array(Y), dtype=torch.long)

        if len(train_idx) == 0 or len(val_idx) == 0:
            return {'fitness': 0.0, 'val_acc': 0.0, 'params': 0, 'flops': 0, 'rf': 0}

        train_loader = DataLoader(Subset(TensorDataset(X_t, Y_t), train_idx),
                                  batch_size=genome.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(Subset(TensorDataset(X_t, Y_t), val_idx),
                                batch_size=256, shuffle=False, pin_memory=True)

        model = build_model_from_genome(genome, in_channels=X_t.shape[1],
                                        num_classes=int(Y_t.max().item()) + 1,
                                        use_dropout=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=genome.lr, weight_decay=genome.wd)
        loss_fn = torch.nn.CrossEntropyLoss()

        best_val_acc = 0.0
        epochs = epochs_full if use_full else epochs_proxy
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
            val_loss, val_acc = evaluate_model_quick_torch(model, device, val_loader)
            best_val_acc = max(best_val_acc, val_acc)
            if verbose:
                print(f"  epoch {epoch+1}/{epochs} val_acc={val_acc:.4f}")

        rf = estimate_receptive_field(genome, input_size=X_t.shape[-1])
        flops, params_est = estimate_flops_params(genome, input_size=X_t.shape[-1], in_channels=X_t.shape[1])
        cfg = fitness_cfg or {}
        fitness = composite_fitness_from_metrics(best_val_acc, params_est, flops, rf,
                                                 input_size=X_t.shape[-1], **cfg)

        return {'fitness': float(fitness), 'val_acc': float(best_val_acc),
                'params': int(params_est), 'flops': int(flops), 'rf': int(rf)}

    except Exception as e:
        warnings.warn(f"fitness error: {e}")
        return {'fitness': 0.0, 'val_acc': 0.0, 'params': 0, 'flops': 0, 'rf': 0}