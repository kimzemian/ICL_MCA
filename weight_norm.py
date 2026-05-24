import torch
import numpy as np

paths = {
    "L16": "/share/dean/mx253/icl_ca/checkpoint/ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-0-d512-lr1e-3-wd0.2-ep500/ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-0-d512-lr1e-3-wd0.2-ep500_best.pt",
    "L32": "/share/dean/mx253/icl_ca/checkpoint/ca-cmixed-L32_T5_M2_seed42_120k-layers2-heads-1-1-mlp1-1-d512-lr1e-3-wd0.2-ep500/ca-cmixed-L32_T5_M2_seed42_120k-layers2-heads-1-1-mlp1-1-d512-lr1e-3-wd0.2-ep500_best.pt",
}

for name, path in paths.items():
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    print(f"\n=== {name} ===")
    total_norm = 0
    for k, v in sorted(state.items()):
        if "positional" in k:
            continue  # skip PE
        n = v.float().norm().item()
        total_norm += n**2
        print(f"  {k}: shape={list(v.shape)}, norm={n:.4f}")
    print(f"  Total norm (excl PE): {np.sqrt(total_norm):.4f}")