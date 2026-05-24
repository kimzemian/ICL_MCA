"""
Autoregressive generation for ECA transformer.

Feeds context rows (0 to M-1), then generates remaining rows token by token.
Sep tokens are inserted manually at known positions.

Usage:
    python ar_generate.py \
        --ckpt /path/to/best.pt \
        --test_path /path/to/mixed_test.npz \
        --cell_width 15 \
        --num_context_rows 4 \
        --num_samples 500
"""

import argparse
import numpy as np
import torch
from data_generate import ECADataset
from models import VSimpleTransformer


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["model_config"]
    if "heads_list" not in cfg:
        num_layers = cfg.get("num_layers", 2)
        num_heads = cfg.get("num_heads", 2)
        cfg["heads_list"] = [num_heads] * num_layers

    pos_emb_weight = ckpt["model_state_dict"]["positional_embedding.weight"]
    seq_len = pos_emb_weight.shape[0]

    model = VSimpleTransformer(
        vocab_size=3,
        hidden_size=cfg["hidden_size"],
        output_size=3,
        seq_len=seq_len,
        heads_list=cfg["heads_list"],
        use_mlp_list=cfg.get("use_mlp_list", None),
        emb_dropout=cfg.get("emb_dropout", 0.0),
        dropout=cfg.get("dropout", 0.0),
    )
    state_dict = ckpt["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("transformer.layers.", "transformer_layers.")] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model, cfg, seq_len


def ar_generate_one(model, context_tokens, total_len, cell_width, device):
    """Autoregressively generate tokens after context.

    Args:
        model: trained model
        context_tokens: 1D tensor of input tokens (context rows + seps)
        total_len: total sequence length to generate (len of input_tokens)
        cell_width: cells per row (excluding sep)
        device: torch device

    Returns:
        generated: 1D numpy array of full sequence (context + generated)
    """
    stride = cell_width + 1
    sep_token = 2

    # Start with context
    tokens = list(context_tokens.numpy())

    with torch.no_grad():
        while len(tokens) < total_len:
            pos_in_row = len(tokens) % stride

            if pos_in_row == cell_width:
                # This position is a separator — insert manually
                tokens.append(sep_token)
            else:
                # Model predicts next token
                x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
                logits = model(x)  # (1, T, 3)
                next_token = logits[0, -1].argmax().item()
                tokens.append(next_token)

    return np.array(tokens[:total_len], dtype=np.int64)


def evaluate_ar(generated, ground_truth_input, ground_truth_target,
                cell_width, num_context_rows):
    """Compare AR generated sequence with ground truth.

    Note: ground_truth is the TARGET sequence (shifted by 1 from input).
    generated is the INPUT sequence. So generated[i] should predict target[i],
    but for AR eval we compare generated cells with the actual cell values.

    We reconstruct the full token sequence from input + last target token,
    then compare row by row.
    """
    stride = cell_width + 1

    # Reconstruct full ground truth token sequence
    # input_tokens = tokens[:-1], target_tokens = tokens[1:]
    # So full tokens = input[0], input[1], ..., input[T-1], target[T-1]
    gt_full = np.concatenate([ground_truth_input.numpy(), [ground_truth_target[-1].item()]])
    gen_full = np.concatenate([generated, [0]])  # pad to same length, last won't be used

    total_tokens = len(gt_full)
    num_rows = total_tokens // stride

    per_row = {}
    for row in range(num_rows):
        if row < num_context_rows:
            continue
        start = row * stride
        end = start + cell_width  # exclude sep
        if end > len(gt_full) or end > len(generated):
            break

        gt_row = gt_full[start:end]
        gen_row = generated[start:end] if end <= len(generated) else gen_full[start:end]

        correct = (gt_row == gen_row).sum()
        total = len(gt_row)
        per_row[row] = {
            "cell_acc": correct / total,
            "zero_error": 1.0 if correct == total else 0.0,
            "correct": int(correct),
            "total": int(total),
        }

    return per_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--cell_width", type=int, required=True)
    parser.add_argument("--num_context_rows", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stride = args.cell_width + 1

    print(f"Loading checkpoint: {args.ckpt}")
    model, cfg, seq_len = load_model(args.ckpt, device)
    print(f"Model config: {cfg}")
    print(f"Seq len: {seq_len}, cell_width: {args.cell_width}, stride: {stride}")

    test_ds = ECADataset(args.test_path)
    indices = np.random.choice(len(test_ds), min(args.num_samples, len(test_ds)), replace=False)

    # Context length: num_context_rows rows + (num_context_rows - 1) seps
    # In input_tokens, context ends at position:
    #   num_context_rows * cell_width + (num_context_rows - 1) seps
    #   = num_context_rows * stride - 1
    context_len = args.num_context_rows * stride - 1

    # Aggregate per-row stats
    row_correct_total = {}
    row_zero_error_total = {}
    row_count = {}

    for i, idx in enumerate(indices):
        if (i + 1) % 100 == 0:
            print(f"  Generating {i+1}/{len(indices)}...")

        X, y, mask = test_ds[idx]
        context = X[:context_len]

        generated = ar_generate_one(model, context, len(X) + 1, args.cell_width, device)

        per_row = evaluate_ar(generated, X, y, args.cell_width, args.num_context_rows)

        for row, stats in per_row.items():
            if row not in row_correct_total:
                row_correct_total[row] = 0
                row_zero_error_total[row] = 0
                row_count[row] = 0
            row_correct_total[row] += stats["correct"]
            row_zero_error_total[row] += stats["zero_error"]
            row_count[row] += 1

    # Print results
    print(f"\n{'='*60}")
    print(f"Autoregressive Generation Results ({len(indices)} samples)")
    print(f"{'='*60}")
    print(f"{'Row':<6} {'Cell Acc':<12} {'Zero-Error':<12} {'Samples'}")
    print(f"{'-'*50}")

    total_correct = 0
    total_cells = 0
    total_zero_error = 0
    total_samples = 0

    for row in sorted(row_correct_total.keys()):
        n = row_count[row]
        cell_acc = row_correct_total[row] / (n * args.cell_width)
        zero_err = row_zero_error_total[row] / n
        print(f"{row:<6} {cell_acc:<12.4f} {zero_err:<12.4f} {n}")

        total_correct += row_correct_total[row]
        total_cells += n * args.cell_width
        total_zero_error += row_zero_error_total[row]
        total_samples += n

    print(f"{'-'*50}")
    overall_acc = total_correct / total_cells if total_cells > 0 else 0
    overall_zer = total_zero_error / total_samples if total_samples > 0 else 0
    num_rows_eval = len(row_correct_total)
    print(f"{'Overall':<6} {overall_acc:<12.4f} {overall_zer:<12.4f}")
    print(f"\nRows evaluated: {num_rows_eval} (row {min(row_correct_total.keys())} to {max(row_correct_total.keys())})")


if __name__ == "__main__":
    main()