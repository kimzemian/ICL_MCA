# ============================================================
# 改动说明：data_generate.py
# ============================================================
# 
# 新增：--L_list 参数，例如 --L_list 9 13 17 21 25
# 每个 sample 随机选一个 L，生成不同宽度的序列
# 所有序列 pad 到最大 L 对应的 seq_len
# PAD token = 3 (vocab: 0, 1, SEP=2, PAD=3)
#
# 用法：
#   python data_generate_mixed_L.py --L_list 9 13 17 21 25 --T 10 \
#       --num_context_rows 4 --require_all_patterns --mix
#
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset
import random
import argparse
import os

# ========== 从原 data_generate.py 复制的部分（不变）==========
CLASS_I = [0, 8, 32, 40, 128, 136, 160, 168]
CLASS_II = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,19,23,24,25,26,27,28,29,
            33,34,35,36,37,38,42,43,44,46,50,51,56,57,58,62,72,73,74,76,
            77,78,94,104,108,130,132,134,138,140,142,152,154,156,162,164,
            170,172,178,184,200,204,232]
CLASS_III = [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150]
CLASS_IV = [41, 54, 106, 110]

EQUIVALENTS = {
    0: [255], 1: [127], 2: [16,191,247], 3: [17,63,119],
    4: [223], 5: [95], 6: [20,159,215], 7: [21,31,87],
    8: [64,239,253], 9: [65,111,125], 10: [80,175,245],
    11: [47,81,117], 12: [68,207,221], 13: [69,79,93],
    14: [84,143,213], 15: [85], 18: [183], 19: [55],
    22: [151], 23: [], 24: [66,189,231], 25: [61,67,103],
    26: [82,167,181], 27: [39,53,83], 28: [70,157,199],
    29: [71], 30: [86,135,149], 32: [251], 33: [123],
    34: [48,187,243], 35: [49,59,115], 36: [219], 37: [91],
    38: [52,155,211], 40: [96,235,249], 41: [97,107,121],
    42: [112,171,241], 43: [113], 44: [100,203,217],
    45: [75,89,101], 46: [116,139,209], 50: [179], 51: [],
    54: [147], 56: [98,185,227], 57: [99], 58: [114,163,177],
    60: [102,153,195], 62: [118,131,145], 72: [237], 73: [109],
    74: [88,173,229], 76: [205], 77: [], 78: [92,141,197],
    90: [165], 94: [133], 104: [233], 105: [],
    106: [120,169,225], 108: [201], 110: [124,137,193],
    122: [161], 126: [129], 128: [254], 130: [144,190,246],
    132: [222], 134: [148,158,214], 136: [192,238,252],
    138: [174,208,224], 140: [196,206,220], 142: [212],
    146: [182], 150: [], 152: [188,194,230], 154: [166,180,210],
    156: [198], 160: [250], 162: [176,186,242], 164: [218],
    168: [224,234,248], 170: [240], 172: [202,216,228],
    178: [], 184: [226], 200: [236], 204: [], 232: [],
}

ALL_CLASSES = {1: CLASS_I, 2: CLASS_II, 3: CLASS_III, 4: CLASS_IV}
PAD_TOKEN = 3


def eca_step(state, rule_binary):
    n = len(state)
    new = np.empty(n, dtype=np.int64)
    for i in range(n):
        pattern = (state[(i - 1) % n] << 2) | (state[i] << 1) | (state[(i + 1) % n])
        new[i] = rule_binary[pattern]
    return new


def rule_to_binary(rule_number):
    return np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.int64)


def get_rule_group(canonical_rule):
    return [canonical_rule] + EQUIVALENTS.get(canonical_rule, [])


def get_patterns_in_row(state):
    n = len(state)
    seen = set()
    for i in range(n):
        pattern = (state[(i - 1) % n] << 2) | (state[i] << 1) | (state[(i + 1) % n])
        seen.add(pattern)
    return seen


def check_context_coverage(states_history, num_context_rows):
    if num_context_rows < 2:
        return False
    all_patterns = set()
    for t in range(num_context_rows - 1):
        all_patterns |= get_patterns_in_row(states_history[t])
    return len(all_patterns) == 8


def split_class_rules(class_rules, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    canonical = list(class_rules)
    rng.shuffle(canonical)
    split_idx = max(1, int(len(canonical) * train_ratio))
    train_rules, test_rules = [], []
    for r in canonical[:split_idx]:
        train_rules.extend(get_rule_group(r))
    for r in canonical[split_idx:]:
        test_rules.extend(get_rule_group(r))
    return sorted(train_rules), sorted(test_rules)


def get_train_test_rules(classes=None, train_ratio=0.8, seed=42):
    if classes is None:
        classes = [1, 2, 3, 4]
    train_rules, test_rules, split_info = [], [], {}
    for c in classes:
        tr, te = split_class_rules(ALL_CLASSES[c], train_ratio, seed)
        train_rules.extend(tr)
        test_rules.extend(te)
        split_info[c] = {"train": tr, "test": te}
    return train_rules, test_rules, split_info


def generate_eca_sequence(grid_size, num_steps, rule_number, sep_token=2, num_context_rows=2,
                          rng=None, require_all_patterns=False, max_attempts=1000):
    """Same as original — returns variable-length arrays."""
    rule_bin = rule_to_binary(rule_number)
    if rng is None:
        rng = np.random.RandomState()

    for attempt in range(max_attempts):
        state = rng.randint(0, 2, size=grid_size).astype(np.int64)
        states_history = [state.copy()]
        cur = state.copy()
        for t in range(1, num_steps):
            cur = eca_step(cur, rule_bin)
            states_history.append(cur.copy())

        if require_all_patterns:
            if not check_context_coverage(states_history, num_context_rows):
                continue

        tokens = []
        step_indices = []
        for t in range(num_steps):
            for cell in states_history[t]:
                tokens.append(cell)
                step_indices.append(t)
            if t < num_steps - 1:
                tokens.append(sep_token)
                step_indices.append(-1)

        tokens = np.array(tokens, dtype=np.int64)
        step_indices = np.array(step_indices, dtype=np.int64)

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        target_steps = step_indices[1:]

        loss_mask = ((target_steps >= num_context_rows) & (target_steps != -1)).astype(np.float32)

        return input_tokens, target_tokens, loss_mask

    raise RuntimeError(
        f"Failed after {max_attempts} attempts (grid_size={grid_size}, rule={rule_number})."
    )


# ========== 新增：Mixed-L 生成 ==========

def compute_seq_len(grid_size, T):
    """Sequence length for input (= total tokens - 1)."""
    return grid_size * T + (T - 1) - 1


def generate_and_save_mixed_L(rules, num_samples, save_path, L_list, T=10, seed=42,
                               num_context_rows=2, require_all_patterns=False):
    """Generate dataset with mixed grid widths. Pad to max seq_len."""
    rng = np.random.RandomState(seed)

    # grid_size = L - 1 for each L
    grid_sizes = [L - 1 for L in L_list]
    max_grid = max(grid_sizes)
    max_seq_len = compute_seq_len(max_grid, T)

    all_inp = np.full((num_samples, max_seq_len), PAD_TOKEN, dtype=np.int64)
    all_tgt = np.full((num_samples, max_seq_len), PAD_TOKEN, dtype=np.int64)
    all_mask = np.zeros((num_samples, max_seq_len), dtype=np.float32)
    all_rules = np.empty(num_samples, dtype=np.int64)
    all_Ls = np.empty(num_samples, dtype=np.int64)
    all_lengths = np.empty(num_samples, dtype=np.int64)  # actual seq length (before pad)

    for i in range(num_samples):
        rule = rules[rng.randint(len(rules))]
        L = L_list[rng.randint(len(L_list))]
        grid_size = L - 1
        sample_rng = np.random.RandomState(rng.randint(0, 2**31))

        inp, tgt, mask = generate_eca_sequence(
            grid_size, T, rule, num_context_rows=num_context_rows,
            rng=sample_rng, require_all_patterns=require_all_patterns,
        )

        actual_len = len(inp)
        all_inp[i, :actual_len] = inp
        all_tgt[i, :actual_len] = tgt
        all_mask[i, :actual_len] = mask
        all_rules[i] = rule
        all_Ls[i] = L
        all_lengths[i] = actual_len

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(save_path,
             inputs=all_inp, targets=all_tgt, masks=all_mask,
             rules=all_rules, Ls=all_Ls, lengths=all_lengths)
    
    L_str = ",".join(str(l) for l in L_list)
    print(f"Saved {num_samples} samples to {save_path} "
          f"(L_list=[{L_str}], T={T}, max_seq_len={max_seq_len}, "
          f"all_patterns={require_all_patterns})")


class ECADatasetMixedL(Dataset):
    """Dataset that also returns actual sequence length for padding mask."""
    def __init__(self, path):
        data = np.load(path)
        self.inputs = data["inputs"]
        self.targets = data["targets"]
        self.masks = data["masks"]
        self.rules = data["rules"] if "rules" in data else None
        self.Ls = data["Ls"] if "Ls" in data else None
        self.lengths = data["lengths"] if "lengths" in data else None
        print(f"Loaded {len(self.inputs)} samples from {path}")
        if self.Ls is not None:
            unique_Ls, counts = np.unique(self.Ls, return_counts=True)
            for L, c in zip(unique_Ls, counts):
                print(f"  L={L}: {c} samples ({100*c/len(self.inputs):.1f}%)")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        length = self.lengths[idx] if self.lengths is not None else len(self.inputs[idx])
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(length, dtype=torch.long),
        )


def count_str(n):
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train", type=int, default=30000)
    parser.add_argument("--num_test", type=int, default=5000)
    parser.add_argument("--L_list", type=int, nargs="+", required=True,
                        help="List of grid widths, e.g. --L_list 9 13 17 21 25")
    parser.add_argument("--L_test_unseen", type=int, nargs="*", default=[],
                        help="Unseen L values for test only, e.g. --L_test_unseen 11 15 19")
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--num_context_rows", type=int, default=4)
    parser.add_argument("--require_all_patterns", action="store_true")
    args = parser.parse_args()

    # Auto name
    L_str = "-".join(str(l) for l in sorted(args.L_list))
    total = args.num_train * len(args.classes)
    if args.output_dir is None:
        args.output_dir = (f"/share/dean/mx253/icl_ca/eca_data/mixedL_{L_str}_T{args.T}_M{args.num_context_rows}"
                           f"_seed{args.seed}_{count_str(total)}")
    os.makedirs(args.output_dir, exist_ok=True)

    _, _, info = get_train_test_rules(args.classes, seed=args.seed)

    print("=== Rule Split ===")
    for c in sorted(info.keys()):
        tr, te = info[c]["train"], info[c]["test"]
        print(f"Class {c}: {len(tr)} train rules, {len(te)} test rules")

    print(f"\n=== Config: L_list={args.L_list}, T={args.T}, "
          f"M={args.num_context_rows}, require_all_patterns={args.require_all_patterns} ===\n")

    # Collect all rules
    all_train_rules, all_test_rules = [], []
    for c in sorted(info.keys()):
        all_train_rules.extend(info[c]["train"])
        all_test_rules.extend(info[c]["test"])

    # Filename helpers: embed L values in the name
    train_L_tag = "L" + "-".join(str(l) for l in sorted(args.L_list))
    seen_L_tag = train_L_tag
    unseen_L_tag = "L" + "-".join(str(l) for l in sorted(args.L_test_unseen)) if args.L_test_unseen else ""

    train_name = f"mixed_train_{train_L_tag}.npz"
    test_seen_name = f"mixed_test_seen_{seen_L_tag}.npz"
    test_unseen_name = f"mixed_test_unseen_{unseen_L_tag}.npz" if args.L_test_unseen else None

    # === Train: mixed L from L_list ===
    generate_and_save_mixed_L(
        all_train_rules, args.num_train * len(args.classes),
        os.path.join(args.output_dir, train_name),
        L_list=args.L_list, T=args.T, seed=args.seed,
        num_context_rows=args.num_context_rows,
        require_all_patterns=args.require_all_patterns,
    )

    # === Test on seen L values ===
    generate_and_save_mixed_L(
        all_test_rules, args.num_test * len(args.classes),
        os.path.join(args.output_dir, test_seen_name),
        L_list=args.L_list, T=args.T, seed=args.seed + 100,
        num_context_rows=args.num_context_rows,
        require_all_patterns=args.require_all_patterns,
    )

    # === Test on unseen L values ===
    if args.L_test_unseen:
        generate_and_save_mixed_L(
            all_test_rules, args.num_test * len(args.classes),
            os.path.join(args.output_dir, test_unseen_name),
            L_list=args.L_test_unseen, T=args.T, seed=args.seed + 200,
            num_context_rows=args.num_context_rows,
            require_all_patterns=args.require_all_patterns,
        )
        print(f"\nNote: unseen test max_seq_len may differ from train. "
              f"Model seq_len must accommodate max(L_list + L_test_unseen).")

    print("\nDone!")
    print(f"Output dir: {args.output_dir}")
    print(f"\nFiles:")
    print(f"  Train:          {args.output_dir}/{train_name}")
    print(f"  Test (seen L):  {args.output_dir}/{test_seen_name}")
    if test_unseen_name:
        print(f"  Test (unseen L): {args.output_dir}/{test_unseen_name}")