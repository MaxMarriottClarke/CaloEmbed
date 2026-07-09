"""Train a node-embedding GNN on HGCal layer clusters.

Usage (from CaloEmbed/training):
  python train.py --config configs/default.yaml
  python train.py --config configs/default.yaml --max-files 20 --epochs 5
  python train.py --config configs/default.yaml --resume

Model and loss are chosen by name in the config (see src/models, src/losses).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from src.data import HDF5Events, discover_files, split_files
from src.loop import run_epoch
from src.losses import build_loss
from src.models import build_model
from src.utils import count_parameters, git_hash, set_seed


def build_scheduler(optimizer, cfg: dict, epochs: int):
    name = cfg.get("scheduler", "none")
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.get("step_size", 25), gamma=cfg.get("gamma", 0.5))
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unknown scheduler '{name}' (none | step | cosine)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="CaloEmbed GNN embedding training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", help="Override data.dir")
    parser.add_argument("--output", help="Override output.dir")
    parser.add_argument("--epochs", type=int, help="Override train.epochs")
    parser.add_argument("--batch-size", type=int, help="Override data.batch_size")
    parser.add_argument("--lr", type=float, help="Override optim.lr")
    parser.add_argument("--max-files", type=int, help="Override data.max_files")
    parser.add_argument("--num-workers", type=int, help="Override data.num_workers")
    parser.add_argument("--device", help="cuda | cpu (default: cuda if available)")
    parser.add_argument("--resume", action="store_true", help="Resume from <output>/last.pt")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data:            cfg["data"]["dir"] = args.data
    if args.output:          cfg["output"]["dir"] = args.output
    if args.epochs:          cfg["train"]["epochs"] = args.epochs
    if args.batch_size:      cfg["data"]["batch_size"] = args.batch_size
    if args.lr:              cfg["optim"]["lr"] = args.lr
    if args.max_files:       cfg["data"]["max_files"] = args.max_files
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    output_dir = Path(cfg["output"]["dir"])
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    if last_path.exists() and not args.resume:
        raise SystemExit(
            f"{last_path} already exists. Use --resume to continue, or a fresh --output.")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    # ── data ─────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    files = discover_files(data_cfg["dir"], data_cfg.get("max_files", -1))
    train_files, val_files, test_files = split_files(
        files, data_cfg.get("val_fraction", 0.1), data_cfg.get("test_fraction", 0.0), seed)
    (output_dir / "splits.json").write_text(json.dumps(
        {k: sorted(p.name for p in v) for k, v in
         [("train", train_files), ("val", val_files), ("test", test_files)]}, indent=2))
    features = data_cfg.get("features", ["x", "y", "z", "energy", "eta", "phi"])
    data_train = HDF5Events(train_files, features, data_cfg.get("max_events", -1))
    data_val = HDF5Events(val_files, features, data_cfg.get("max_events", -1))
    print(f"Data: {len(data_train)} train / {len(data_val)} val events "
          f"({len(train_files)}/{len(val_files)}/{len(test_files)} train/val/test files, "
          f"test never loaded), features={features}")

    loader_kwargs = dict(
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(data_train, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(data_val, shuffle=False, **loader_kwargs)

    # ── model / loss / optim ─────────────────────────────────────────────
    model = build_model({**cfg["model"], "in_dim": len(features)}).to(device)
    loss_fn = build_loss(cfg["loss"])
    print(f"Model: {cfg['model']['name']} ({count_parameters(model):,} params) "
          f"| loss: {cfg['loss']['name']} | device: {device}")

    optim_cfg = cfg["optim"]
    epochs = cfg["train"]["epochs"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=optim_cfg["lr"],
        weight_decay=optim_cfg.get("weight_decay", 0.0))
    scheduler = build_scheduler(optimizer, optim_cfg, epochs)
    amp = cfg["train"].get("amp", True)
    grad_clip = cfg["train"].get("grad_clip")
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_val = state["epoch"] + 1, state["best_val"]
        print(f"Resumed from {last_path} at epoch {start_epoch} (best val {best_val:.6f})")

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    csv_path = output_dir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])

    # ── loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, train_loader, loss_fn, device,
                               optimizer=optimizer, scaler=scaler, amp=amp,
                               grad_clip=grad_clip, desc=f"epoch {epoch + 1} train")
        val_loss = run_epoch(model, val_loader, loss_fn, device, amp=amp,
                             desc=f"epoch {epoch + 1} val")
        lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()
        seconds = time.perf_counter() - t0

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "config": cfg,
        }, last_path)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, lr, round(seconds, 1)])
        print(f"epoch {epoch + 1}/{epochs}  train {train_loss:.6f}  val {val_loss:.6f}"
              f"  lr {lr:.2e}  ({seconds:.0f}s){'  *best*' if is_best else ''}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_hash": git_hash(),
        "config": cfg,
        "n_train_events": len(data_train),
        "n_val_events": len(data_val),
        "best_val_loss": best_val,
        "epochs_run": epochs,
        "checkpoints": {"best": str(best_path), "last": str(last_path)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. best val {best_val:.6f} → {best_path}")


if __name__ == "__main__":
    main()
