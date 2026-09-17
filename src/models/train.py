"""Trains ST-SAGE on the synthetic India district-month crime panel and
saves a checkpoint + training curve."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CRIME_CATEGORIES, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED
from src.models.dataset import make_datasets
from src.models.st_sage import STSage, negative_binomial_nll


def _collate_with_adjacency(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == "t":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def run_epoch(model, loader, adj_norm, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            mu, log_disp = model(batch, adj_norm)
            loss = negative_binomial_nll(mu, log_disp, batch["target"])
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main(epochs: int = 25, batch_size: int = 8, lr: float = 1e-3, device: str = "cpu"):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Building datasets (this runs the causal SAE smoothing pass)...")
    train_ds, val_ds, test_ds, graph, global_log_rate = make_datasets()
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} months; "
          f"{graph.n_districts} districts")

    adj_norm = torch.from_numpy(graph.adjacency_norm).float().to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_with_adjacency)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_with_adjacency)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_with_adjacency)

    model = STSage(n_categories=len(CRIME_CATEGORIES), global_log_rate=global_log_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, adj_norm, optimizer, device)
        val_loss = run_epoch(model, val_loader, adj_norm, None, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch:02d}  train_nll={train_loss:.4f}  val_nll={val_loss:.4f}  "
              f"({time.time()-t0:.1f}s)")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = run_epoch(model, test_loader, adj_norm, None, device)
    print(f"final test_nll={test_loss:.4f}")

    ckpt_path = MODELS_DIR / "st_sage.pt"
    torch.save({"state_dict": model.state_dict(), "n_categories": len(CRIME_CATEGORIES)}, ckpt_path)
    with open(OUTPUTS_DIR / "training_history.json", "w") as f:
        json.dump({**history, "test_nll": test_loss}, f, indent=2)
    print(f"Saved checkpoint to {ckpt_path}")
    return model, adj_norm, (train_ds, val_ds, test_ds, graph)


if __name__ == "__main__":
    main()
