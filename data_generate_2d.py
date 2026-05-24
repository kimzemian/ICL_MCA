"""
2D Cellular Automata data generation (Von Neumann neighborhood, k=5, V=2).

Fixes over original:
  1. Balanced sampling: uniformly sample equivalence class first, then rule within class
  2. Separate val set (same rules as test, different samples)
  3. Distribution stats printed after generation
  4. Dedup check
  5. [NEW] Algorithmic complexity filtering (zlib compressibility)
  6. [NEW] Sparse variable initial conditions (10% to 50% density)
  7. [NEW] Skewed Lambda bin sampling (heavy focus on transition bins)

Usage:
    python data_generate_2d.py --L1 6 --L2 6 --T 12 --num_context_rows 8 \
        --num_train 500000 --num_val 20000 --num_test 20000 \
        --num_rules_train 200 --num_rules_test 50 \
        --require_all_patterns --seed 42
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import random
import argparse
import os
import zlib
from collections import Counter


# ============================================================
# Von Neumann neighborhood: k=5
# Offsets: N=(-1,0), W=(0,-1), C=(0,0), E=(0,1), S=(1,0)
# ============================================================
VN_OFFSETS = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
K = 5
V = 2
NUM_CONFIGS = V**K  # 32
NUM_RULES = V**NUM_CONFIGS  # 2^32

# Tokens
TOK_0 = 0
TOK_1 = 1
TIME_SEP = 2
ROW_SEP = 3
VOCAB_SIZE = 4


# ============================================================
# Symmetry group for Von Neumann neighborhood
# D4 (8 elements) x complement (2 elements) = 16 transformations
# ============================================================


def _offset_index(offset):
    return VN_OFFSETS.index(offset)


def _apply_transform_to_offset(offset, transform):
    r, c = offset
    if transform == "id":
        return (r, c)
    elif transform == "r90":
        return (c, -r)
    elif transform == "r180":
        return (-r, -c)
    elif transform == "r270":
        return (-c, r)
    elif transform == "fh":
        return (r, -c)
    elif transform == "fv":
        return (-r, c)
    elif transform == "fd":
        return (c, r)
    elif transform == "fa":
        return (-c, -r)
    raise ValueError(f"Unknown transform: {transform}")


D4_TRANSFORMS = ["id", "r90", "r180", "r270", "fh", "fv", "fd", "fa"]


def _build_permutation(transform):
    perm = []
    for i, offset in enumerate(VN_OFFSETS):
        new_offset = _apply_transform_to_offset(offset, transform)
        perm.append(_offset_index(new_offset))
    return perm


D4_PERMS = {t: _build_permutation(t) for t in D4_TRANSFORMS}


def _config_to_index(config):
    idx = 0
    for bit in config:
        idx = idx * 2 + bit
    return idx


def _index_to_config(idx, k=K):
    config = []
    for _ in range(k):
        config.append(idx % 2)
        idx //= 2
    return tuple(reversed(config))


def transform_rule(rule_int, transform, complement=False):
    perm = D4_PERMS[transform]
    new_rule = 0
    for cfg_idx in range(NUM_CONFIGS):
        config = _index_to_config(cfg_idx)
        if complement:
            config = tuple(1 - x for x in config)
        permuted_config = tuple(config[perm[j]] for j in range(K))
        permuted_idx = _config_to_index(permuted_config)
        output = (rule_int >> permuted_idx) & 1
        if complement:
            output = 1 - output
        new_rule |= output << cfg_idx
    return new_rule


def canonical_rule(rule_int):
    min_rule = rule_int
    for transform in D4_TRANSFORMS:
        for complement in [False, True]:
            transformed = transform_rule(rule_int, transform, complement)
            min_rule = min(min_rule, transformed)
    return min_rule


def get_equivalence_class(rule_int):
    equiv = set()
    for transform in D4_TRANSFORMS:
        for complement in [False, True]:
            equiv.add(transform_rule(rule_int, transform, complement))
    return equiv


# ============================================================
# 2D CA simulation
# ============================================================


def ca2d_step(grid, rule_int, offsets=VN_OFFSETS):
    L1, L2 = grid.shape
    new_grid = np.empty_like(grid)
    for r in range(L1):
        for c in range(L2):
            config_idx = 0
            for dr, dc in offsets:
                nr = (r + dr) % L1
                nc = (c + dc) % L2
                config_idx = config_idx * 2 + int(grid[nr, nc])
            new_grid[r, c] = (rule_int >> config_idx) & 1
    return new_grid


def get_patterns_in_grid(grid, offsets=VN_OFFSETS):
    L1, L2 = grid.shape
    seen = set()
    for r in range(L1):
        for c in range(L2):
            config_idx = 0
            for dr, dc in offsets:
                nr = (r + dr) % L1
                nc = (c + dc) % L2
                config_idx = config_idx * 2 + int(grid[nr, nc])
            seen.add(config_idx)
    return seen


def check_context_coverage_2d(grids, num_context_rows):
    if num_context_rows < 2:
        return False
    all_patterns = set()
    for t in range(num_context_rows - 1):
        all_patterns |= get_patterns_in_grid(grids[t])
    return len(all_patterns) == NUM_CONFIGS


# ============================================================
# Tokenization
# ============================================================


def flatten_trajectory(grids, num_steps, use_row_sep=True):
    tokens = []
    step_indices = []
    L1, L2 = grids[0].shape

    for t in range(num_steps):
        for r in range(L1):
            for c in range(L2):
                tokens.append(int(grids[t][r, c]))
                step_indices.append(t)
            if use_row_sep and r < L1 - 1:
                tokens.append(ROW_SEP)
                step_indices.append(t)
        if t < num_steps - 1:
            tokens.append(TIME_SEP)
            step_indices.append(t)

    return np.array(tokens, dtype=np.int64), np.array(step_indices, dtype=np.int64)


def compute_seq_len(L1, L2, T, use_row_sep=True):
    if use_row_sep:
        tokens_per_step = L1 * L2 + (L1 - 1)
    else:
        tokens_per_step = L1 * L2
    total_tokens = T * tokens_per_step + (T - 1)
    return total_tokens - 1


# ============================================================
# Data generation with balanced class sampling
# ============================================================


def build_class_to_rules(rules):
    """Group rules by their canonical (equivalence class) representative."""
    class_to_rules = {}
    for r in rules:
        canon = canonical_rule(r)
        if canon not in class_to_rules:
            class_to_rules[canon] = []
        class_to_rules[canon].append(r)
    class_list = sorted(class_to_rules.keys())
    return class_to_rules, class_list


def build_bin_to_rules(rules):
    """Group rules by their lambda bin."""
    bin_to_rules = {b: [] for b in range(len(LAMBDA_BINS))}
    for r in rules:
        b = get_lambda_bin(r)
        bin_to_rules[b].append(r)
    return bin_to_rules


def sample_rule_lambda_balanced(bin_to_rules, rng):
    """Uniformly sample a lambda bin, then uniformly sample a rule within it."""
    non_empty_bins = [b for b, rules in bin_to_rules.items() if len(rules) > 0]
    b = non_empty_bins[rng.randint(len(non_empty_bins))]
    rules_in_bin = bin_to_rules[b]
    return rules_in_bin[rng.randint(len(rules_in_bin))]


def is_complex_enough(grids):
    """Check if the sequence is neither trivial nor pure noise via Zlib compression."""
    raw_bytes = np.array(grids).tobytes()
    compressed = zlib.compress(raw_bytes)
    ratio = len(compressed) / len(raw_bytes)
    # Trivial < 0.05 (highly compressible)
    # Pure noise > 0.85 (hardly compressible)
    return 0.05 < ratio < 0.85


def generate_2d_sequence(
    L1,
    L2,
    num_steps,
    rule_int,
    num_context_rows=4,
    rng=None,
    require_all_patterns=False,
    max_attempts=1000,
    use_row_sep=True,
):
    if rng is None:
        rng = np.random.RandomState()

    for attempt in range(max_attempts):
        # Force sparse variable initialization
        p_active = rng.uniform(0.1, 0.5)
        grid = rng.choice([0, 1], size=(L1, L2), p=[1 - p_active, p_active]).astype(
            np.int64
        )

        grids = [grid.copy()]
        cur = grid.copy()
        for t in range(1, num_steps):
            cur = ca2d_step(cur, rule_int)
            grids.append(cur.copy())

        # Filter out sequences that are dead or too chaotic
        if not is_complex_enough(grids):
            continue

        if require_all_patterns:
            if not check_context_coverage_2d(grids, num_context_rows):
                continue

        tokens, step_indices = flatten_trajectory(grids, num_steps, use_row_sep)

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        target_steps = step_indices[1:]

        loss_mask = (target_steps >= num_context_rows).astype(np.float32)

        return input_tokens, target_tokens, loss_mask, attempt + 1

    raise RuntimeError(
        f"Failed to generate valid sample after {max_attempts} attempts "
        f"(L1={L1}, L2={L2}, rule={rule_int}). "
        f"Rule may be too trivial or grid too small for all {NUM_CONFIGS} patterns to appear."
    )


def generate_and_save_2d(
    rules,
    num_samples,
    save_path,
    L1=16,
    L2=16,
    T=8,
    seed=42,
    num_context_rows=4,
    require_all_patterns=False,
    use_row_sep=True,
):
    """Generate dataset with lambda-balanced sampling."""
    rng = np.random.RandomState(seed)
    seq_len = compute_seq_len(L1, L2, T, use_row_sep)

    class_to_rules, class_list = build_class_to_rules(rules)
    bin_to_rules = build_bin_to_rules(rules)

    for b, (lo, hi) in enumerate(LAMBDA_BINS):
        print(
            f"    Bin {b} [λ∈[{lo:.2f},{hi:.2f})]: {len(bin_to_rules[b])} rules available"
        )

    all_inp = np.empty((num_samples, seq_len), dtype=np.int64)
    all_tgt = np.empty((num_samples, seq_len), dtype=np.int64)
    all_mask = np.empty((num_samples, seq_len), dtype=np.float32)
    all_rules = np.empty(num_samples, dtype=np.int64)

    total_attempts = 0
    rule_counter = Counter()
    class_counter = Counter()

    for i in range(num_samples):
        if (i + 1) % 10000 == 0:
            avg_attempts = total_attempts / (i + 1)
            print(
                f"  Generating sample {i+1}/{num_samples}... "
                f"(avg attempts/sample: {avg_attempts:.1f})"
            )

        max_rule_retries = (
            20  # Increased to allow for stricter complex filter rejection
        )
        for retry in range(max_rule_retries):
            rule = sample_rule_lambda_balanced(bin_to_rules, rng)
            sample_rng = np.random.RandomState(rng.randint(0, 2**31))
            try:
                inp, tgt, mask, attempts = generate_2d_sequence(
                    L1,
                    L2,
                    T,
                    rule,
                    num_context_rows=num_context_rows,
                    rng=sample_rng,
                    require_all_patterns=require_all_patterns,
                    max_attempts=2000,
                    use_row_sep=use_row_sep,
                )
                break
            except RuntimeError:
                if retry == max_rule_retries - 1:
                    raise RuntimeError(
                        f"Failed to generate sample {i} after {max_rule_retries} rule retries"
                    )
                continue

        all_inp[i] = inp
        all_tgt[i] = tgt
        all_mask[i] = mask
        all_rules[i] = rule
        total_attempts += attempts
        rule_counter[rule] += 1
        class_counter[canonical_rule(rule)] += 1

    print(f"\n  === Distribution Stats ===")
    print(f"  Total equivalence classes: {len(class_list)}")
    print(f"  Unique rules sampled: {len(rule_counter)}")

    class_counts = list(class_counter.values())
    print(
        f"  Samples per class: min={min(class_counts)}, max={max(class_counts)}, "
        f"mean={np.mean(class_counts):.1f}, std={np.std(class_counts):.1f}"
    )

    class_sizes = [len(class_to_rules[c]) for c in class_list]
    print(
        f"  Rules per class: min={min(class_sizes)}, max={max(class_sizes)}, "
        f"mean={np.mean(class_sizes):.1f}"
    )

    if require_all_patterns:
        avg_attempts = total_attempts / num_samples
        print(f"  Avg rejection attempts per sample: {avg_attempts:.1f}")

    classes_seen = set(canonical_rule(int(r)) for r in all_rules)
    missing_classes = set(class_list) - classes_seen
    print(f"  Equiv classes represented: {len(classes_seen)}/{len(class_list)}")
    if missing_classes:
        print(f"  WARNING: {len(missing_classes)} classes have ZERO samples!")

    rules_seen = set(int(r) for r in all_rules)
    all_possible_rules = set(rules)
    missing_rules = all_possible_rules - rules_seen
    print(
        f"  Individual rules represented: {len(rules_seen)}/{len(all_possible_rules)}"
    )
    if missing_rules:
        print(f"  WARNING: {len(missing_rules)} rules have ZERO samples!")

    lambda_bins_count = [0] * len(LAMBDA_BINS)
    for r in all_rules:
        b = get_lambda_bin(int(r))
        lambda_bins_count[b] += 1
    print(f"  Lambda distribution in generated data:")
    for b, (lo, hi) in enumerate(LAMBDA_BINS):
        print(
            f"    Bin {b} [λ∈[{lo:.2f},{hi:.2f})]: {lambda_bins_count[b]} samples "
            f"({100*lambda_bins_count[b]/num_samples:.1f}%)"
        )

    hashes = set()
    num_dupes = 0
    for i in range(num_samples):
        h = hash(all_inp[i].tobytes())
        if h in hashes:
            num_dupes += 1
        hashes.add(h)
    print(
        f"  Duplicate input sequences: {num_dupes}/{num_samples} "
        f"({100*num_dupes/num_samples:.2f}%)"
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(
        save_path, inputs=all_inp, targets=all_tgt, masks=all_mask, rules=all_rules
    )
    vocab = 4 if use_row_sep else 3
    print(
        f"\n  Saved {num_samples} samples to {save_path} "
        f"(grid={L1}x{L2}, T={T}, seq_len={seq_len}, vocab={vocab}, "
        f"use_row_sep={use_row_sep}, num_classes={len(class_list)}, "
        f"all_patterns={require_all_patterns})"
    )


def get_lambda(rule_int):
    """Langton's lambda: fraction of configs that map to state 1."""
    return bin(rule_int).count("1") / NUM_CONFIGS


LAMBDA_BINS = [(0.0, 0.25), (0.25, 0.45), (0.45, 0.55), (0.55, 1.0)]


def get_lambda_bin(rule_int):
    """Return which lambda bin (0-3) this rule falls into."""
    lam = get_lambda(rule_int)
    for i, (lo, hi) in enumerate(LAMBDA_BINS):
        if lo <= lam < hi:
            return i
    return len(LAMBDA_BINS) - 1


def rule_can_cover(
    rule_int, L1, L2, num_context_rows, num_tries=100, min_successes=3, seed=None
):
    """Pre-screen: can this rule reliably produce all 32 configs in context rows?"""
    rng = np.random.RandomState(seed)
    successes = 0
    for _ in range(num_tries):
        # Apply sparse initialization to match generation criteria
        p_active = rng.uniform(0.1, 0.5)
        grid = rng.choice([0, 1], size=(L1, L2), p=[1 - p_active, p_active]).astype(
            np.int64
        )

        grids = [grid.copy()]
        cur = grid.copy()
        for t in range(1, num_context_rows):
            cur = ca2d_step(cur, rule_int)
            grids.append(cur.copy())

        if check_context_coverage_2d(grids, num_context_rows):
            successes += 1
            if successes >= min_successes:
                return True
    return False


def sample_rules_stratified(num_rules, seed=42, L1=6, L2=6, num_context_rows=8):
    """Sample rules heavily skewing toward the 'edge of chaos' bins."""
    num_bins = len(LAMBDA_BINS)

    # Force ~10% to extremes, ~40% to the middle transition zones
    bin_quotas = [
        int(num_rules * 0.10),  # Bin 0: [0.0, 0.25)
        int(num_rules * 0.40),  # Bin 1: [0.25, 0.45)
        int(num_rules * 0.40),  # Bin 2: [0.45, 0.55)
        int(num_rules * 0.10),  # Bin 3: [0.55, 1.0]
    ]
    # Handle any remainder from rounding
    bin_quotas[2] += num_rules - sum(bin_quotas)

    rng = random.Random(seed)
    canonical_seen = set()
    bin_rules = [[] for _ in range(num_bins)]
    bin_done = [False] * num_bins
    rejected_by_coverage = 0

    max_iters = num_rules * 5000
    iters = 0
    while not all(bin_done) and iters < max_iters:
        iters += 1
        rule = rng.randint(0, NUM_RULES - 1)
        canon = canonical_rule(rule)
        if canon in canonical_seen:
            continue

        b = get_lambda_bin(rule)
        if len(bin_rules[b]) >= bin_quotas[b]:
            bin_done[b] = True
            continue

        # Pre-screen: can this rule produce all 32 configs?
        if not rule_can_cover(
            rule, L1, L2, num_context_rows, num_tries=20, seed=rng.randint(0, 2**31)
        ):
            rejected_by_coverage += 1
            canonical_seen.add(canon)
            continue

        canonical_seen.add(canon)
        bin_rules[b].append(rule)
        if len(bin_rules[b]) >= bin_quotas[b]:
            bin_done[b] = True

    rules = []
    for b in range(num_bins):
        rules.extend(bin_rules[b])

    print(f"  Lambda-skewed sampling ({num_rules} rules across {num_bins} bins):")
    print(f"  Rules rejected by coverage pre-screen: {rejected_by_coverage}")
    for b in range(num_bins):
        lo, hi = LAMBDA_BINS[b]
        lambdas = [get_lambda(r) for r in bin_rules[b]]
        if lambdas:
            print(
                f"    Bin {b} [λ∈[{lo:.2f},{hi:.2f})]: {len(bin_rules[b])}/{bin_quotas[b]} rules, "
                f"λ mean={np.mean(lambdas):.3f}"
            )
        else:
            print(
                f"    Bin {b} [λ∈[{lo:.2f},{hi:.2f})]: 0/{bin_quotas[b]} rules (EMPTY!)"
            )

    if not all(bin_done):
        unfilled = [b for b in range(num_bins) if len(bin_rules[b]) < bin_quotas[b]]
        print(
            f"  WARNING: Could not fill bins {unfilled}. "
            f"These λ ranges may have too few rules that can produce all 32 configs."
        )

    return rules, canonical_seen


def split_rules(
    num_train_rules, num_test_rules, seed=42, L1=6, L2=6, num_context_rows=8
):
    """Sample train and test rules with non-overlapping equivalence classes."""
    total = num_train_rules + num_test_rules
    all_rules, all_canonicals = sample_rules_stratified(
        total, seed=seed, L1=L1, L2=L2, num_context_rows=num_context_rows
    )

    rng = random.Random(seed + 1)
    num_bins = len(LAMBDA_BINS)

    bin_to_rules = [[] for _ in range(num_bins)]
    for r in all_rules:
        bin_to_rules[get_lambda_bin(r)].append(r)

    train_rules_raw = []
    test_rules_raw = []

    train_frac = num_train_rules / total
    for b in range(num_bins):
        bin_list = bin_to_rules[b]
        rng.shuffle(bin_list)
        n_train = round(len(bin_list) * train_frac)
        train_rules_raw.extend(bin_list[:n_train])
        test_rules_raw.extend(bin_list[n_train:])

    train_rules = []
    for r in train_rules_raw:
        train_rules.extend(get_equivalence_class(r))
    test_rules = []
    for r in test_rules_raw:
        test_rules.extend(get_equivalence_class(r))

    train_rules = sorted(set(train_rules))
    test_rules = sorted(set(test_rules))

    assert len(set(train_rules) & set(test_rules)) == 0, "Train/test overlap!"

    print(f"\n  Train/test split by lambda bin:")
    for b in range(num_bins):
        lo, hi = LAMBDA_BINS[b]
        n_tr = sum(1 for r in train_rules_raw if get_lambda_bin(r) == b)
        n_te = sum(1 for r in test_rules_raw if get_lambda_bin(r) == b)
        print(f"    Bin {b} [λ∈[{lo:.2f},{hi:.2f})]: train={n_tr}, test={n_te}")

    return train_rules, test_rules


class CA2DDataset(Dataset):
    """Dataset for 2D CA sequences."""

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


# ============================================================
# Main
# ============================================================


def count_str(n):
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--L1", type=int, default=16, help="Grid height")
    parser.add_argument("--L2", type=int, default=16, help="Grid width")
    parser.add_argument("--T", type=int, default=8, help="Number of time steps")
    parser.add_argument("--num_context_rows", type=int, default=4)
    parser.add_argument("--num_train", type=int, default=30000)
    parser.add_argument(
        "--num_val",
        type=int,
        default=5000,
        help="Number of validation samples (uses test rules)",
    )
    parser.add_argument("--num_test", type=int, default=5000)
    parser.add_argument("--num_rules_train", type=int, default=200)
    parser.add_argument("--num_rules_test", type=int, default=50)
    parser.add_argument("--require_all_patterns", action="store_true")
    parser.add_argument("--no_row_sep", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_row_sep = not args.no_row_sep
    vocab = 4 if use_row_sep else 3
    sep_label = "ROW_SEP+TIME_SEP" if use_row_sep else "TIME_SEP only"

    if args.output_dir is None:
        sep_tag = "rowsep" if use_row_sep else "nosep"
        args.output_dir = (
            f"./eca_data/"
            f"VN_L{args.L1}x{args.L2}_T{args.T}_M{args.num_context_rows}"
            f"_rules{args.num_rules_train}+{args.num_rules_test}_aug"
            f"_{sep_tag}_seed{args.seed}_{count_str(args.num_train)}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== 2D CA Data Generation (Von Neumann, k={K}) ===")
    print(f"Grid: {args.L1}x{args.L2}, T={args.T}, M={args.num_context_rows}")
    print(f"Configs: V^k = {NUM_CONFIGS}, Total rules: 2^{NUM_CONFIGS}")
    print(f"Separators: {sep_label} (vocab={vocab})")
    print(
        f"Sampling {args.num_rules_train} train + {args.num_rules_test} test equiv classes"
    )
    print(f"Samples: {args.num_train} train, {args.num_val} val, {args.num_test} test")
    print(f"require_all_patterns: {args.require_all_patterns}")
    print(f"Sequence length: {compute_seq_len(args.L1, args.L2, args.T, use_row_sep)}")
    print()

    # Sample rules
    print("Sampling rules...")
    train_rules, test_rules = split_rules(
        args.num_rules_train,
        args.num_rules_test,
        seed=args.seed,
        L1=args.L1,
        L2=args.L2,
        num_context_rows=args.num_context_rows,
    )

    train_class_to_rules, train_class_list = build_class_to_rules(train_rules)
    test_class_to_rules, test_class_list = build_class_to_rules(test_rules)

    print(
        f"  Train: {len(train_rules)} rules from {len(train_class_list)} equiv classes"
    )
    print(
        f"    Class sizes: {[len(train_class_to_rules[c]) for c in train_class_list[:5]]}... "
    )
    print(f"  Test:  {len(test_rules)} rules from {len(test_class_list)} equiv classes")
    print(
        f"    Class sizes: {[len(test_class_to_rules[c]) for c in test_class_list[:5]]}... "
    )

    train_canon = set(canonical_rule(r) for r in train_rules)
    test_canon = set(canonical_rule(r) for r in test_rules)
    print(f"  Overlap: {len(train_canon & test_canon)} (should be 0)")
    print()

    # Save rule lists
    np.savez(
        os.path.join(args.output_dir, "rules.npz"),
        train_rules=np.array(train_rules, dtype=np.int64),
        test_rules=np.array(test_rules, dtype=np.int64),
    )

    # Generate training data
    print("=" * 50)
    print("Generating training data...")
    generate_and_save_2d(
        train_rules,
        args.num_train,
        os.path.join(args.output_dir, "train.npz"),
        L1=args.L1,
        L2=args.L2,
        T=args.T,
        seed=args.seed,
        num_context_rows=args.num_context_rows,
        require_all_patterns=args.require_all_patterns,
        use_row_sep=use_row_sep,
    )

    # Generate validation data (same rules as test, different seed)
    print("\n" + "=" * 50)
    print("Generating validation data (test rules, different samples)...")
    generate_and_save_2d(
        test_rules,
        args.num_val,
        os.path.join(args.output_dir, "val.npz"),
        L1=args.L1,
        L2=args.L2,
        T=args.T,
        seed=args.seed + 50,
        num_context_rows=args.num_context_rows,
        require_all_patterns=args.require_all_patterns,
        use_row_sep=use_row_sep,
    )

    # Generate test data (same rules as val, different seed)
    print("\n" + "=" * 50)
    print("Generating test data...")
    generate_and_save_2d(
        test_rules,
        args.num_test,
        os.path.join(args.output_dir, "test.npz"),
        L1=args.L1,
        L2=args.L2,
        T=args.T,
        seed=args.seed + 100,
        num_context_rows=args.num_context_rows,
        require_all_patterns=args.require_all_patterns,
        use_row_sep=use_row_sep,
    )

    print(f"\nDone! Output dir: {args.output_dir}")
    print(f"Files: train.npz, val.npz, test.npz, rules.npz")
