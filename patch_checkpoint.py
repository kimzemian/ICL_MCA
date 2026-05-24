"""
Patch old checkpoints that are missing use_mlp_list in model_config.
Infers use_mlp_list from the run name (e.g. mlp1-0 → [True, False]).

Usage:
    python patch_checkpoints.py
"""

import torch
import os
import re
import glob

CKPT_DIR = "/share/dean/mx253/icl_ca/checkpoint"

# Find all runs with mlp in the name
runs = glob.glob(os.path.join(CKPT_DIR, "ca-*mlp*"))

for run_dir in sorted(runs):
    run_name = os.path.basename(run_dir)

    # Extract mlp config from name, e.g. "mlp1-0" → [True, False]
    match = re.search(r"mlp([01](?:-[01])*)", run_name)
    if not match:
        print(f"SKIP (can't parse mlp): {run_name}")
        continue

    mlp_str = match.group(1)
    use_mlp_list = [bool(int(x)) for x in mlp_str.split("-")]
    
    # Skip if already all True (default behavior is correct)
    if all(use_mlp_list):
        print(f"SKIP (all True): {run_name}")
        continue

    # Patch all .pt files in this directory
    pt_files = glob.glob(os.path.join(run_dir, "*.pt"))
    for pt_path in pt_files:
        try:
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            if "model_config" in ckpt:
                if "use_mlp_list" not in ckpt["model_config"] or ckpt["model_config"]["use_mlp_list"] is None:
                    ckpt["model_config"]["use_mlp_list"] = use_mlp_list
                    torch.save(ckpt, pt_path)
                    print(f"PATCHED: {os.path.basename(pt_path)} → use_mlp_list={use_mlp_list}")
                else:
                    print(f"OK (already has): {os.path.basename(pt_path)}")
        except Exception as e:
            print(f"ERROR: {os.path.basename(pt_path)}: {e}")