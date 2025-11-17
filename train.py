import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from data import load_sampled_data
from utils import print_classification_metrics
# from models.transformer_model_1 import TransformerModel
from models.hierarchial_transformer import HierarchicalTransformerEncoder
from models.selfgated_hierarchial_transformer import (
    SelfGatedHierarchicalTransformerEncoder
)


FIX_SEED = 2025


def set_seed(seed: int = FIX_SEED) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (may impact performance)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def gates_to_sensor_segment_matrix(extras, reduce='max'):
    """
    Convert per-d_model gates into per-sensor × segment weights.

    Approx attribution: weight[f, s] ≈ ∑_d |W_proj[f,d]| * gate[s,d]
    Where W_proj is the input projection (d_model x F in PyTorch param shape).

    Args:
        extras: dict from forward(return_gates=True)
                - 'gates': (B, S, d_model)
                - 'W_proj': (d_model, F) torch.Tensor
        reduce: 'mean' or 'max' across batch dimension

    Returns:
        M: (F, S) numpy array
    """
    gates = extras['gates']     
    W_proj = extras['W_proj']   

    
    if isinstance(gates, torch.Tensor):
        G = gates.detach().cpu().numpy()
    else:
        G = np.asarray(gates)
    G = G.max(axis=0) if reduce == 'max' else G.mean(axis=0)  

    # |W_proj|^T -> (F, d_model)
    if isinstance(W_proj, torch.Tensor):
        Wabs_T = torch.abs(W_proj).T.detach().cpu().numpy()
    else:
        Wabs_T = np.abs(np.asarray(W_proj)).T

 
    M = Wabs_T @ G.T
    return M


def plot_gating_heatmap(M, sensor_names=None, fault_id=None, out_png="gating_heatmap_test.png",
                        title_prefix="Gating Weights (Sensors × Segments)"):
    F, S = M.shape
    vmax = np.max(np.abs(M)) + 1e-12
    
    if np.min(M) < 0:
        vmin, vmax_plot = -vmax, +vmax
        cmap_kwargs = dict(vmin=vmin, vmax=vmax_plot)
    else:
        cmap_kwargs = {}

    if sensor_names is None:
        sensor_names = [f"Var{i+1}" for i in range(F)]

    plt.figure(figsize=(max(6, 0.6*S), max(6, 0.25*F)))
    plt.imshow(M, aspect="auto", interpolation="nearest", **cmap_kwargs)
    cbar_label = "Gate Weight" if np.min(M) >= 0 else "Δ Gate Weight (fault − baseline)"
    plt.colorbar(label=cbar_label)
    plt.xticks(np.arange(S), [f"Seg {i}" for i in range(S)])
    plt.yticks(np.arange(F), sensor_names)
    title = title_prefix
    if fault_id is not None:
        title += f" – Fault {fault_id}"
    plt.title(title)
    plt.xlabel("Segments")
    plt.ylabel("Sensors")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_topk_sensors(M, sensor_names=None, k=10, fault_id=None, out_png="gating_top10_test.png",
                      label="Mean Gate Weight (across segments)", title_prefix="Top Sensors"):
    F, S = M.shape
    if sensor_names is None:
        sensor_names = [f"Var{i+1}" for i in range(F)]

    mean_per_sensor = M.mean(axis=1)  
    idx = np.argsort(-mean_per_sensor)[:k]
    vals = mean_per_sensor[idx][::-1]
    names = [sensor_names[i] for i in idx][::-1]

    plt.figure(figsize=(8, max(4, 0.4*k + 2)))
    plt.barh(np.arange(len(vals)), vals)
    plt.yticks(np.arange(len(vals)), names)
    plt.xlabel(label)
    title = title_prefix
    if fault_id is not None:
        title += f" – Fault {fault_id}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def _compute_M_for_indices(model, X_tensor, idx_slice, reduce='max'):
    """Forward with gates on X_tensor[idx_slice] and return sensor×segment M."""
    with torch.no_grad():
        logits, extras = model(X_tensor[idx_slice], return_gates=True)
    M = gates_to_sensor_segment_matrix(extras, reduce=reduce)  # (F, S)
    return M


def make_fault_gating_plots_with_delta(model, X_test_tensor, y_test_tensor,
                                       fault_ids, baseline_fault=0,
                                       k_top=10, n_windows=128, reduce='max'):
    """
    For each fault in fault_ids, save:
      - absolute heatmap & top-k
      - delta (fault - baseline fault 0) heatmap & top-k
    """
    os.makedirs("figures", exist_ok=True)
    model.eval()

    idx0 = (y_test_tensor == baseline_fault).nonzero(as_tuple=False).squeeze(-1)
    M0 = None
    if idx0.numel() == 0:
        print(f"[WARN] No windows for baseline fault {baseline_fault}. Delta plots will be skipped.")
    else:
        sel0 = idx0[:min(n_windows, idx0.numel())]
        print(f"[BASELINE] fault={baseline_fault}, windows used={sel0.numel()}")
        M0 = _compute_M_for_indices(model, X_test_tensor, sel0, reduce=reduce)

    with torch.no_grad():
        for fid in fault_ids:
            idx = (y_test_tensor == fid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                print(f"[WARN] No test windows found for fault {fid}; skipping.")
                continue

            sel = idx[:min(n_windows, idx.numel())]
            print(f"[FAULT] fid={fid}, windows used={sel.numel()}")

            M = _compute_M_for_indices(model, X_test_tensor, sel, reduce=reduce)
            sensor_names = [f"Var{i+1}" for i in range(M.shape[0])]

            # Absolute plots
            plot_gating_heatmap(
                M, sensor_names, fault_id=fid,
                out_png=f"figures/gating_heatmap_fault_{fid}.png",
                title_prefix="Gating Weights (Sensors × Segments)"
            )
            plot_topk_sensors(
                M, sensor_names, k=k_top, fault_id=fid,
                out_png=f"figures/gating_top{k_top}_fault_{fid}.png",
                label="Mean Gate Weight (across segments)",
                title_prefix="Top Sensors by Gate Weight"
            )

            # Delta vs baseline (if available)
            if M0 is not None and M0.shape == M.shape:
                Md = M - M0  # signed differences
                # Heatmap (signed, symmetric color range set inside)
                plot_gating_heatmap(
                    Md, sensor_names, fault_id=fid,
                    out_png=f"figures/gating_delta_heatmap_fault_{fid}.png",
                    title_prefix="Δ Gating (fault − baseline)"
                )
               
                plot_topk_sensors(
                    Md, sensor_names, k=k_top, fault_id=fid,
                    out_png=f"figures/gating_delta_top{k_top}_fault_{fid}.png",
                    label="Mean Δ Gate Weight (vs baseline)",
                    title_prefix="Top Sensors by Δ Gate"
                )

                print(f"[INFO] Saved fault {fid} plots (abs + delta).")
            else:
                print(f"[INFO] Baseline missing or shape mismatch; saved absolute plots for fault {fid} only.")


# ---------------- TRAIN / EVAL ----------------

def train_model(model, X_train, y_train, X_test, y_test, num_classes, device):
    """
    Trains a Transformer-based model on time-series data using mixed precision,
    and evaluates it on a test set.
    """
    model = model.to(device)

    # tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.long).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()

    num_epochs = 50
    batch_size = 128
    loss_history = []

    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            optimizer.zero_grad()
            if device.type == "cuda":
                with autocast(device_type="cuda"):
                    output = model(batch_x)
                    loss = criterion(output, batch_y.squeeze())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(batch_x)
                loss = criterion(output, batch_y.squeeze())
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, (X_train_tensor.size(0) // batch_size))
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), "transformer_model_1.pth")

    # Loss curve
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("loss_curve.png")
    plt.close()

    model.eval()
    eval_batch_size = 256  # you can tune this (128, 256, 512, etc.)

    all_preds = []

    with torch.no_grad():
        for i in range(0, X_test_tensor.size(0), eval_batch_size):
            batch_x = X_test_tensor[i : i + eval_batch_size]
            logits = model(batch_x)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)

        y_pred = torch.cat(all_preds).numpy()
        y_true = y_test_tensor.cpu().numpy()

        print_classification_metrics(y_true, y_pred)

        # Global (mixed faults) gating plots (absolute only)
        slice_idx = slice(0, min(128, X_test_tensor.size(0)))
        logits, extras = model(X_test_tensor[slice_idx], return_gates=True)
        M_global = gates_to_sensor_segment_matrix(extras, reduce='max')  # (F, S)
        sensor_names = [f"Var{i+1}" for i in range(M_global.shape[0])]
        os.makedirs("figures", exist_ok=True)
        plot_gating_heatmap(M_global, sensor_names, fault_id=None,
                            out_png="figures/gating_heatmap_test.png",
                            title_prefix="Gating Weights (Sensors × Segments)")
        plot_topk_sensors(M_global, sensor_names, k=10, fault_id=None,
                          out_png="figures/gating_top10_test.png",
                          label="Mean Gate Weight (across segments)",
                          title_prefix="Top Sensors by Gate Weight")
        print("[INFO] Saved global gating plots to figures/: gating_heatmap_test.png, gating_top10_test.png")

        make_fault_gating_plots_with_delta(
            model,
            X_test_tensor,
            y_test_tensor,
            fault_ids=[3, 9, 15],   
            baseline_fault=0,     
            k_top=10,
            n_windows=512,
            reduce='max'          
        )

def main():
    """
    Load data, instantiate the model, train, evaluate, and save gating plots.
    """
    # Fix random seed for reproducibility
    set_seed()

    # Load data (your settings)
    X_train, X_test, y_train, y_test = load_sampled_data(window_size=100, stride=5)

    print("Training Data Shape:", X_train.shape, y_train.shape)
    print("Test Data Shape:", X_test.shape, y_test.shape)
    print("Unique classes in Training Set:", np.unique(y_train))
    print("Unique classes in Test Set:", np.unique(y_test))
    num_classes = 21

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SelfGatedHierarchicalTransformerEncoder(
        input_dim=X_train.shape[2],
        d_model=128,
        nhead=4,
        num_layers_low=2,
        num_layers_high=2,
        dim_feedforward=128,
        dropout=0.05,
        pool_output_size=10,
        num_classes=num_classes
    )

    train_model(model, X_train, y_train, X_test, y_test, num_classes, device)


if __name__ == "__main__":
    main()
