# All 4 classes, per-class files + mixed (30k train per class, default)
# python data_generate.py --L 16 --T 10 --num_context_rows 4 --require_all_patterns --mix

# All 4 classes, larger dataset (100k train per class)
# python data_generate.py --L 64 --T 8 --num_context_rows 4 --num_train 100000 --num_test 10000 --require_all_patterns --mix

# Single rule only
# python data_generate.py --single_rule 110 --L 16 --T 10 --num_context_rows 4 --require_all_patterns

# Subset of classes (no mixed file unless --mix is added)
# python data_generate.py --L 64 --T 4 --num_context_rows 2 --classes 3 4 --mix
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import argparse
import os

# ========== Rule Classes ==========
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


def eca_step(state, rule_binary):
    """One step of ECA with periodic boundary conditions."""
    n = len(state)
    new = np.empty(n, dtype=np.int64)
    for i in range(n):
        pattern = (state[(i - 1) % n] << 2) | (state[i] << 1) | (state[(i + 1) % n])
        new[i] = rule_binary[pattern]
    return new


def rule_to_binary(rule_number):
    """Convert rule number (0-255) to 8-bit lookup table."""
    return np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.int64)


def get_rule_group(canonical_rule):
    """Get canonical + all equivalent rules."""
    return [canonical_rule] + EQUIVALENTS.get(canonical_rule, [])


def get_patterns_in_row(state):
    """Get the set of 3-bit patterns present in a row (with periodic BC)."""
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
        f"Failed to generate valid sample after {max_attempts} attempts "
        f"(grid_size={grid_size}, rule={rule_number}). "
        f"Grid may be too small for all 8 patterns to appear in context rows."
    )


def generate_and_save(rules, num_samples, save_path, L=64, T=4, seed=42, num_context_rows=2,
                      require_all_patterns=False):
    grid_size = L - 1
    rng = np.random.RandomState(seed)

    seq_len = grid_size * T + (T - 1) - 1
    all_inp = np.empty((num_samples, seq_len), dtype=np.int64)
    all_tgt = np.empty((num_samples, seq_len), dtype=np.int64)
    all_mask = np.empty((num_samples, seq_len), dtype=np.float32)
    all_rules = np.empty(num_samples, dtype=np.int64)

    for i in range(num_samples):
        rule = rules[rng.randint(len(rules))]
        sample_rng = np.random.RandomState(rng.randint(0, 2**31))
        inp, tgt, mask = generate_eca_sequence(
            grid_size, T, rule, num_context_rows=num_context_rows,
            rng=sample_rng, require_all_patterns=require_all_patterns,
        )
        all_inp[i] = inp
        all_tgt[i] = tgt
        all_mask[i] = mask
        all_rules[i] = rule

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(save_path, inputs=all_inp, targets=all_tgt, masks=all_mask, rules=all_rules)
    print(f"Saved {num_samples} samples to {save_path} "
          f"(grid={grid_size}, T={T}, seq_len={seq_len}, "
          f"all_patterns={require_all_patterns})")


class ECADataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.inputs = data["inputs"]
        self.targets = data["targets"]
        self.masks = data["masks"]
        self.rules = data["rules"] if "rules" in data else None
        print(f"Loaded {len(self.inputs)} samples from {path}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.float32),
        )


def count_str(n):
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train", type=int, default=30000,
                        help="Number of train samples (per class in multi-class mode)")
    parser.add_argument("--num_test", type=int, default=5000,
                        help="Number of test samples (per class in multi-class mode)")
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Which classes to generate (1-4)")
    parser.add_argument("--num_context_rows", type=int, default=2,
                        help="Number of free context rows (no loss). Must be >= 2 for pattern+output.")
    parser.add_argument("--require_all_patterns", action="store_true",
                        help="Reject samples where context rows don't cover all 8 patterns")
    parser.add_argument("--mix", action="store_true",
                        help="Also generate a mixed file with all classes combined")
    parser.add_argument("--single_rule", type=int, default=None,
                        help="Generate data for a single rule number (0-255)")
    args = parser.parse_args()

    if args.single_rule is not None:
        # === Single rule mode ===
        rule = args.single_rule
        if args.output_dir is None:
            args.output_dir = (f"/share/dean/mx253/icl_ca/eca_data/"
                               f"L{args.L}_T{args.T}_M{args.num_context_rows}_seed{args.seed}_rule{rule}")
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"\n=== Single rule mode: rule {rule} ===")
        print(f"Output dir: {args.output_dir}\n")

        generate_and_save(
            [rule], args.num_train,
            os.path.join(args.output_dir, f"rule{rule}_train.npz"),
            L=args.L, T=args.T, seed=args.seed, num_context_rows=args.num_context_rows,
            require_all_patterns=args.require_all_patterns,
        )
        generate_and_save(
            [rule], args.num_test,
            os.path.join(args.output_dir, f"rule{rule}_test.npz"),
            L=args.L, T=args.T, seed=args.seed + 100, num_context_rows=args.num_context_rows,
            require_all_patterns=args.require_all_patterns,
        )

    else:
        # === Multi-class mode ===
        if args.output_dir is None:
            total = args.num_train * len(args.classes)
            args.output_dir = (f"/share/dean/mx253/icl_ca/eca_data/"
                               f"L{args.L}_T{args.T}_M{args.num_context_rows}_seed{args.seed}_{count_str(total)}")
        os.makedirs(args.output_dir, exist_ok=True)

        _, _, info = get_train_test_rules(args.classes, seed=args.seed)

        print("=== Rule Split ===")
        for c in sorted(info.keys()):
            tr, te = info[c]["train"], info[c]["test"]
            print(f"Class {c}: {len(tr)} train rules, {len(te)} test rules")

        print(f"\n=== Config: L={args.L} (grid={args.L-1}), T={args.T}, "
              f"num_context_rows={args.num_context_rows}, require_all_patterns={args.require_all_patterns} ===\n")

        for c in sorted(info.keys()):
            generate_and_save(
                info[c]["train"], args.num_train,
                os.path.join(args.output_dir, f"class{c}_train.npz"),
                L=args.L, T=args.T, seed=args.seed + c, num_context_rows=args.num_context_rows,
                require_all_patterns=args.require_all_patterns,
            )
            generate_and_save(
                info[c]["test"], args.num_test,
                os.path.join(args.output_dir, f"class{c}_test.npz"),
                L=args.L, T=args.T, seed=args.seed + c + 100, num_context_rows=args.num_context_rows,
                require_all_patterns=args.require_all_patterns,
            )

        if args.mix:
            all_train_rules = []
            all_test_rules = []
            for c in sorted(info.keys()):
                all_train_rules.extend(info[c]["train"])
                all_test_rules.extend(info[c]["test"])
            print(f"\n=== Generating mixed dataset: {len(all_train_rules)} train rules, "
                  f"{len(all_test_rules)} test rules ===\n")
            generate_and_save(
                all_train_rules, args.num_train * len(args.classes),
                os.path.join(args.output_dir, "mixed_train.npz"),
                L=args.L, T=args.T, seed=args.seed + 99, num_context_rows=args.num_context_rows,
                require_all_patterns=args.require_all_patterns,
            )
            generate_and_save(
                all_test_rules, args.num_test * len(args.classes),
                os.path.join(args.output_dir, "mixed_test.npz"),
                L=args.L, T=args.T, seed=args.seed + 199, num_context_rows=args.num_context_rows,
                require_all_patterns=args.require_all_patterns,
            )