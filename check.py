import numpy as np
import os

dirs = [
    "/share/dean/mx253/icl_ca/eca_data/L16_T10_M4_seed42_30k",
    "/share/dean/mx253/icl_ca/eca_data/L10_T20_M8_seed42_120k",
    "/share/dean/mx253/icl_ca/eca_data/L32_T5_M2_seed42_120k",
]

for d in dirs:
    print(f"\n=== {os.path.basename(d)} ===")
    for fname in sorted(os.listdir(d)):
        if fname.endswith(".npz"):
            data = np.load(os.path.join(d, fname))
            n = len(data["inputs"])
            seq_len = data["inputs"].shape[1]
            print(f"  {fname}: {n} samples, seq_len={seq_len}")