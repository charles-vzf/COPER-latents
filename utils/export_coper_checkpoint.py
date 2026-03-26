#!/usr/bin/env python3
"""
Export a COPER training checkpoint (.ckpt from utils/run_exp.py) to a standard PyTorch bundle.

Output:
  <out_dir>/<name>.pt   — torch.save dict with state_dict + meta
  <out_dir>/<name>.json — human-readable metadata
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--dataset", type=str, default="mimic")
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--drop", type=float, default=None)
    p.add_argument("--random-seed", type=int, default=2022)
    p.add_argument("--num-latents", type=int, default=48)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--rec-layers", type=int, default=3)
    p.add_argument("--units", type=int, default=128)
    p.add_argument("--emb-dim", type=int, default=32)
    p.add_argument("--self-per-cross-attn", type=int, default=1)
    p.add_argument("--latent-heads", type=int, default=2)
    p.add_argument("--cross-heads", type=int, default=1)
    p.add_argument("--cross-dim-head", type=int, default=128)
    p.add_argument("--latent-dim-head", type=int, default=128)
    p.add_argument("--ff-dropout", type=float, default=0.5)
    p.add_argument("--att-dropout", type=float, default=0.5)
    p.add_argument("--ode-dropout", type=float, default=0.5)
    p.add_argument(
        "--second-node",
        action="store_true",
        help="Metadata flag: model was trained with --second-node (2-NODE COPER).",
    )
    p.add_argument("--copy-raw-ckpt", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    ckpt = args.ckpt.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    state = torch.load(ckpt, map_location="cpu")

    meta = {
        "format_version": 1,
        "framework": "pytorch",
        "model_architecture": "COPER",
        "repo_path": str(repo),
        "source_checkpoint": str(ckpt),
        "task": "mimic_in_hospital_mortality",
        "hyperparams": {
            "model_type": "COPER",
            "dataset": args.dataset,
            "fold": args.fold,
            "drop": args.drop,
            "random_seed": args.random_seed,
            "num_latents": args.num_latents,
            "latent_dim": args.latent_dim,
            "rec_layers": args.rec_layers,
            "units": args.units,
            "emb_dim": args.emb_dim,
            "self_per_cross_attn": args.self_per_cross_attn,
            "latent_heads": args.latent_heads,
            "cross_heads": args.cross_heads,
            "cross_dim_head": args.cross_dim_head,
            "latent_dim_head": args.latent_dim_head,
            "ff_dropout": args.ff_dropout,
            "att_dropout": args.att_dropout,
            "ode_dropout": args.ode_dropout,
            "cont_in": True,
            "cont_out": False,
            "second_node": bool(args.second_node),
            "seq_len": 48,
            "input_size_mimic": 76,
        },
        "embedding_definition": {
            "last_latent": "LayerNorm output at latent index -1, shape (batch, latent_dim)",
            "latent_grid": "Full Perceiver latents after LayerNorm, shape (batch, num_latents, latent_dim)",
        },
    }

    bundle = {
        "format_version": 1,
        "model_state_dict": state,
        "meta": meta,
    }

    pt_path = out_dir / f"{args.name}.pt"
    json_path = out_dir / f"{args.name}.json"
    torch.save(bundle, pt_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    if args.copy_raw_ckpt:
        raw_dst = out_dir / f"{args.name}_original.ckpt"
        shutil.copy2(ckpt, raw_dst)

    print("Saved:", pt_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
