"""
1D Cellular Automata data generation with configurable V states, k=3 neighbors.

Usage:
    # V=3
    python data_generate_1d_general.py --V 3 --L 32 --T 12 --num_context_rows 8 \
        --require_all_patterns --num_train 120000 --num_test 20000 \
        --num_rules_train 200 --num_rules_test 50

    # V=2 (recovers standard ECA)
    python data_generate_1d_general.py --V 2 --L 16 --T 10 --num_context_rows 4 \
        --require_all_patterns --num_train 120000 --num_test 20000 \
        --num_rules_train 200 --num_rules_test 50

    # V=4
    python data_generate_1d_general.py --V 4 --L 48 --T 10 --num_context_rows 6 \
        --require_all_patterns --num_train 120000 --num_test 20000 \
        --num_rules_train 200 --num_rules_test 50
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import random
import argparse
import os
from itertools import permutations
import math


K = 3  # 1D neighborhood size (left, center, right)


# ============================================================
# Rule representation: base-V number with V^K digits
# ============================================================

def config_to_index(config, V):
    idx = 0
    for c in config:
        idx = idx * V + c
    return idx


def index_to_config(idx, V):
    config = []
    for _ in range(K):
        config.append(idx % V)
        idx //= V
    return tuple(reversed(config))


def rule_lookup(rule_int, config_idx, V):
    return (rule_int // (V ** config_idx)) % V


def rule_from_table(table, V):
    rule_int = 0
    for i, out in enumerate(table):
        rule_int += out * (V ** i)
    return rule_int


def rule_to_table(rule_int, V):
    num_configs = V ** K
    table = []
    r = rule_int
    for _ in range(num_configs):
        table.append(r % V)
        r //= V
    return table


# ============================================================
# Symmetry: reflection × S_V (all permutations of state space)
# ============================================================

def get_state_permutations(V):
    return list(permutations(range(V)))


def transform_rule(rule_int, V, reflect=False, state_perm=None):
    if state_perm is None:
        state_perm = tuple(range(V))

    inv_perm = [0] * V
    for i, p in enumerate(state_perm):
        inv_perm[p] = i

    num_configs = V ** K
    new_table = [0] * num_configs

    for cfg_idx in range(num_configs):
        config = index_to_config(cfg_idx, V)

        # Apply inverse state permutation to input
        mapped_config = tuple(inv_perm[c] for c in config)

        # Apply reflection
        if reflect:
            mapped_config = (mapped_config[2], mapped_config[1], mapped_config[0])

        mapped_idx = config_to_index(mapped_config, V)
        output = rule_lookup(rule_int, mapped_idx, V)

        # Apply state permutation to output
        new_output = state_perm[output]
        new_table[cfg_idx] = new_output

    return rule_from_table(new_table, V)


def canonical_rule(rule_int, V):
    min_rule = rule_int
    for reflect in [False, True]:
        for perm in get_state_permutations(V):
            transformed = transform_rule(rule_int, V, reflect, perm)
            min_rule = min(min_rule, transformed)
    return min_rule


def get_equivalence_class(rule_int, V):
    equiv = set()
    for reflect in [False, True]:
        for perm in get_state_permutations(V):
            equiv.add(transform_rule(rule_int, V, reflect, perm))
    return equiv


# ============================================================
# 1D CA simulation
# ============================================================

def ca_step(state, rule_int, V):
    n = len(state)
    new = np.empty(n, dtype=np.int64)
    for i in range(n):
        config_idx = config_to_index((
            int(state[(i - 1) % n]),
            int(state[i]),
            int(state[(i + 1) % n]),
        ), V)
        new[i] = rule_lookup(rule_int, config_idx, V)
    return new


def get_patterns_in_row(state, V):
    n = len(state)
    seen = set()
    for i in range(n):
        config_idx = config_to_index((
            int(state[(i - 1) % n]),
            int(state[i]),
            int(state[(i + 1) % n]),
        ), V)
        seen.add(config_idx)
    return seen


def check_context_coverage(states_history, num_context_rows, V):
    if num_context_rows < 2:
        return False
    num_configs = V ** K
    all_patterns = set()
    for t in range(num_context_rows - 1):
        all_patterns |= get_patterns_in_row(states_history[t], V)
    return len(all_patterns) == num_configs


# ============================================================
# Sequence generation
# ============================================================

def generate_sequence(grid_size, num_steps, rule_int, V, num_context_rows=8,
                      rng=None, require_all_patterns=False, max_attempts=5000):
    sep_token = V  # SEP = V
    if rng is None:
        rng = np.random.RandomState()

    for attempt in range(max_attempts):
        state = rng.randint(0, V, size=grid_size).astype(np.int64)

        states_history = [state.copy()]
        cur = state.copy()
        for t in range(1, num_steps):
            cur = ca_step(cur, rule_int, V)
            states_history.append(cur.copy())

        if require_all_patterns:
            if not check_context_coverage(states_history, num_context_rows, V):
                continue

        tokens = []
        step_indices = []
        for t in range(num_steps):
            for cell in states_history[t]:
                tokens.append(int(cell))
                step_indices.append(t)
            if t < num_steps - 1:
                tokens.append(sep_token)
                step_indices.append(t)

        tokens = np.array(tokens, dtype=np.int64)
        step_indices = np.array(step_indices, dtype=np.int64)

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        target_steps = step_indices[1:]

        # Loss mask: all tokens in prediction rows (including SEP)
        loss_mask = (target_steps >= num_context_rows).astype(np.float32)

        return input_tokens, target_tokens, loss_mask

    raise RuntimeError(
        f"Failed after {max_attempts} attempts "
        f"(grid_size={grid_size}, V={V}, rule={rule_int}). "
        f"Grid may be too small for all {V**K} patterns."
    )


# ============================================================
# Rule sampling
# ============================================================

def sample_rules(num_rules, V, seed=42):
    rng = random.Random(seed)
    num_configs = V ** K
    canonical_seen = set()
    rules = []

    while len(rules) < num_rules:
        table = [rng.randint(0, V - 1) for _ in range(num_configs)]
        rule = rule_from_table(table, V)
        canon = canonical_rule(rule, V)
        if canon not in canonical_seen:
            canonical_seen.add(canon)
            rules.append(rule)

    return rules, canonical_seen


def split_rules(num_train_rules, num_test_rules, V, seed=42):
    total = num_train_rules + num_test_rules
    all_rules, _ = sample_rules(total, V, seed=seed)

    rng = random.Random(seed + 1)
    indices = list(range(total))
    rng.shuffle(indices)

    # Expand each rule to its full equivalence class
    train_rules = []
    for i in indices[:num_train_rules]:
        train_rules.extend(get_equivalence_class(all_rules[i], V))
    test_rules = []
    for i in indices[num_train_rules:]:
        test_rules.extend(get_equivalence_class(all_rules[i], V))

    train_rules = sorted(set(train_rules))
    test_rules = sorted(set(test_rules))

    assert len(set(train_rules) & set(test_rules)) == 0, "Train/test overlap!"
    return train_rules, test_rules


# ============================================================
# Dataset
# ============================================================

def compute_seq_len(grid_size, T):
    return grid_size * T + (T - 1) - 1


def generate_and_save(rules, num_samples, save_path, V, L=32, T=12, seed=42,
                      num_context_rows=8, require_all_patterns=False):
    grid_size = L - 1
    rng = np.random.RandomState(seed)
    seq_len = compute_seq_len(grid_size, T)

    all_inp = np.empty((num_samples, seq_len), dtype=np.int64)
    all_tgt = np.empty((num_samples, seq_len), dtype=np.int64)
    all_mask = np.empty((num_samples, seq_len), dtype=np.float32)
    # Use object array for rules if they don't fit int64
    num_configs = V ** K
    max_rule = V ** num_configs
    use_int64 = max_rule < 2**63
    if use_int64:
        all_rules = np.empty(num_samples, dtype=np.int64)
    else:
        all_rules = np.empty(num_samples, dtype=object)

    for i in range(num_samples):
        if (i + 1) % 5000 == 0:
            print(f"  Generating sample {i+1}/{num_samples}...")

        rule = rules[rng.randint(len(rules))]
        sample_rng = np.random.RandomState(rng.randint(0, 2**31))

        inp, tgt, mask = generate_sequence(
            grid_size, T, rule, V,
            num_context_rows=num_context_rows,
            rng=sample_rng,
            require_all_patterns=require_all_patterns,
        )

        all_inp[i] = inp
        all_tgt[i] = tgt
        all_mask[i] = mask
        all_rules[i] = rule

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(save_path,
             inputs=all_inp, targets=all_tgt, masks=all_mask, rules=all_rules)
    vocab_size = V + 1
    print(f"Saved {num_samples} samples to {save_path} "
          f"(V={V}, grid={grid_size}, T={T}, seq_len={seq_len}, "
          f"vocab={vocab_size}, configs={V**K}, "
          f"num_rules={len(rules)}, all_patterns={require_all_patterns})")


class CADataset(Dataset):
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
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


# ============================================================
# Coverage bound
# ============================================================

def coverage_bound(V, L, P=0.99):
    """Minimum context transitions for P coverage of all V^K configs."""
    num_configs = V ** K
    grid_size = L - 1
    num = math.log((1 - P) / num_configs)
    den = grid_size * math.log(1 - 1 / num_configs)
    return math.ceil(num / den)


# ============================================================
# Main
# ============================================================

def count_str(n):
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--V", type=int, required=True, help="Number of cell states")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--num_context_rows", type=int, default=8)
    parser.add_argument("--num_train", type=int, default=120000)
    parser.add_argument("--num_test", type=int, default=20000)
    parser.add_argument("--num_rules_train", type=int, default=200)
    parser.add_argument("--num_rules_test", type=int, default=50)
    parser.add_argument("--require_all_patterns", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    V = args.V
    grid_size = args.L - 1
    num_configs = V ** K
    vocab_size = V + 1
    max_symmetry = 2 * math.factorial(V)

    if args.output_dir is None:
        args.output_dir = (
            f"/share/dean/mx253/icl_ca/eca_data/"
            f"V{V}_L{args.L}_T{args.T}_M{args.num_context_rows}"
            f"_rules{args.num_rules_train}+{args.num_rules_test}"
            f"_seed{args.seed}_{count_str(args.num_train)}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    t_min = coverage_bound(V, args.L)

    print(f"=== 1D CA Data Generation ===")
    print(f"V={V}, k={K}, configs={num_configs}")
    print(f"Vocab: {{0..{V-1}, SEP={V}}}, size={vocab_size}")
    print(f"Grid: L={args.L} ({grid_size} cells), T={args.T}, M={args.num_context_rows}")
    print(f"Coverage bound: T_min={t_min} transitions (M={t_min+1} rows) for P>=0.99")
    print(f"Symmetry: reflection x S_{V} = up to {max_symmetry} per class")
    print(f"Sampling {args.num_rules_train} train + {args.num_rules_test} test equiv classes")
    print(f"require_all_patterns: {args.require_all_patterns}")
    print(f"Sequence length: {compute_seq_len(grid_size, args.T)}")
    print()

    if args.num_context_rows < t_min + 1:
        print(f"WARNING: M={args.num_context_rows} < T_min+1={t_min+1}, "
              f"coverage may require many retries!")
        print()

    print("Sampling rules...")
    train_rules, test_rules = split_rules(
        args.num_rules_train, args.num_rules_test, V, seed=args.seed)

    n_train_classes = len(set(canonical_rule(r, V) for r in train_rules))
    n_test_classes = len(set(canonical_rule(r, V) for r in test_rules))
    print(f"  Train: {len(train_rules)} rules ({n_train_classes} equiv classes)")
    print(f"  Test:  {len(test_rules)} rules ({n_test_classes} equiv classes)")
    print(f"  Overlap: {len(set(train_rules) & set(test_rules))} (should be 0)")
    print()

    np.savez(os.path.join(args.output_dir, "rules.npz"),
             train_rules=np.array(train_rules if V <= 3 else [str(r) for r in train_rules]),
             test_rules=np.array(test_rules if V <= 3 else [str(r) for r in test_rules]))

    print("Generating training data...")
    generate_and_save(
        train_rules, args.num_train,
        os.path.join(args.output_dir, "train.npz"),
        V=V, L=args.L, T=args.T, seed=args.seed,
        num_context_rows=args.num_context_rows,
        require_all_patterns=args.require_all_patterns,
    )

    print("Generating test data...")
    generate_and_save(
        test_rules, args.num_test,
        os.path.join(args.output_dir, "test.npz"),
        V=V, L=args.L, T=args.T, seed=args.seed + 100,
        num_context_rows=args.num_context_rows,
        require_all_patterns=args.require_all_patterns,
    )

    print(f"\nDone! Output dir: {args.output_dir}")