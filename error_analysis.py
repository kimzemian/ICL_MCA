"""
Error & success attention analysis for ECA transformer.

Usage:
    python error_analysis.py \
        --ckpt /path/to/best.pt \
        --test_path /path/to/mixed_test.npz \
        --cell_width 15 \
        --show_all_errors \
        --success_per_row 2
"""

import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data_generate import ECADataset
from torch.utils.data import DataLoader, Subset
from models import VSimpleTransformer
from utils import extract_attention_maps, generate_attention_html, count_prior_patterns


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
    return model, cfg


def scan_samples(model, loader, device, cell_width, show_all_errors=False,
                 max_errors=50, success_per_row=2):
    """Scan dataset for error and success cases.

    Returns:
        errors: list of (X, y, error_positions, global_idx, rule)
        successes: dict[row] -> list of (X, y, pos, global_idx, rule)
    """
    stride = cell_width + 1
    model.eval()

    errors = []
    # Per-row buckets for uniform success sampling
    success_by_row = defaultdict(list)

    sample_offset = 0
    with torch.no_grad():
        for X, y, mask in loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            output = model(X)
            preds = output.argmax(dim=-1)
            B, T = X.shape
            eval_mask = mask > 0

            is_error = (preds != y) & eval_mask
            has_error = is_error.any(dim=1)

            for b in range(B):
                global_idx = sample_offset + b

                if has_error[b]:
                    err_pos = is_error[b].nonzero(as_tuple=True)[0].cpu().tolist()
                    if show_all_errors or len(errors) < max_errors:
                        errors.append((X[b].cpu(), y[b].cpu(), err_pos, global_idx))

                else:
                    # Collect success positions grouped by row
                    ok_pos = eval_mask[b].nonzero(as_tuple=True)[0].cpu().tolist()
                    for pos in ok_pos:
                        row = pos // stride
                        success_by_row[row].append((X[b].cpu(), y[b].cpu(), pos, global_idx))

            sample_offset += B

    # Uniformly sample successes: success_per_row from each row
    successes = []
    for row in sorted(success_by_row.keys()):
        candidates = success_by_row[row]
        k = min(success_per_row, len(candidates))
        successes.extend(random.sample(candidates, k))

    return errors, successes


def get_cell_categories(cell_values, q_pos, cell_width, stride):
    """For a query position, return index sets: parent, prior_output, prior_parent."""
    query_row = q_pos // stride
    query_col = q_pos % stride
    actual_col = (query_col + 1) % cell_width

    parent_indices = set()
    prior_output_indices = set()
    prior_parent_indices = set()

    nb_row = query_row - 1
    if nb_row < 0:
        return parent_indices, prior_output_indices, prior_parent_indices

    nb_cols = [(actual_col - 1) % cell_width, actual_col, (actual_col + 1) % cell_width]
    for c in nb_cols:
        parent_indices.add(nb_row * stride + c)

    pattern_bits = []
    for c in nb_cols:
        pattern_bits.append(int(cell_values[nb_row * stride + c]))
    query_pattern = tuple(pattern_bits)

    for r in range(0, query_row - 1):
        for c in range(cell_width):
            left = (c - 1) % cell_width
            right = (c + 1) % cell_width
            l_idx = r * stride + left
            c_idx = r * stride + c
            r_idx = r * stride + right
            if r_idx >= len(cell_values):
                continue
            bits = (int(cell_values[l_idx]), int(cell_values[c_idx]), int(cell_values[r_idx]))
            if bits == query_pattern:
                prior_parent_indices.add(l_idx)
                prior_parent_indices.add(c_idx)
                prior_parent_indices.add(r_idx)
                out_idx = (r + 1) * stride + c
                if out_idx < len(cell_values):
                    prior_output_indices.add(out_idx)

    prior_output_indices -= parent_indices
    prior_parent_indices -= parent_indices
    prior_parent_indices -= prior_output_indices

    return parent_indices, prior_output_indices, prior_parent_indices


def run_attn_distribution(model, test_ds, device, cell_width, num_samples,
                          num_context_rows, seed=42):
    """Compute per layer/head attention distribution over cell categories."""
    np.random.seed(seed)
    stride = cell_width + 1
    indices = np.random.choice(len(test_ds), min(num_samples, len(test_ds)), replace=False)
    subset = Subset(test_ds, indices)

    layer_head_sums = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    layer_head_counts = defaultdict(lambda: defaultdict(int))
    total_cells = 0

    for i, (X, y, mask) in enumerate(subset):
        if (i + 1) % 50 == 0:
            print(f"  Attn distribution: {i+1}/{len(subset)}...")

        attn_maps, preds = extract_attention_maps(model, X.unsqueeze(0), device)
        cell_values = X.numpy()
        T = len(cell_values)

        for pos in range(T):
            if mask[pos].item() == 0:
                continue
            if pos % stride == cell_width:
                continue
            row = pos // stride
            if row < num_context_rows:
                continue

            parent_idx, prior_out_idx, prior_par_idx = get_cell_categories(
                cell_values, pos, cell_width, stride)
            if not parent_idx:
                continue

            for layer_idx, attn in attn_maps.items():
                num_heads = attn.shape[0]
                for h in range(num_heads):
                    h_attn = attn[h, pos, :]
                    parent_attn = sum(h_attn[j] for j in parent_idx)
                    prior_out_attn = sum(h_attn[j] for j in prior_out_idx)
                    prior_par_attn = sum(h_attn[j] for j in prior_par_idx)
                    other_attn = 1.0 - parent_attn - prior_out_attn - prior_par_attn

                    layer_head_sums[layer_idx][h] += np.array([
                        parent_attn, prior_out_attn, prior_par_attn, other_attn])
                    layer_head_counts[layer_idx][h] += 1

            total_cells += 1

    # Compute averages
    categories = ["parent", "prior_output", "prior_parent", "other"]
    results = {}
    for layer_idx in sorted(layer_head_sums.keys()):
        for h in sorted(layer_head_sums[layer_idx].keys()):
            avg = layer_head_sums[layer_idx][h] / layer_head_counts[layer_idx][h]
            results[f"L{layer_idx}_H{h}"] = avg

    # Log to wandb
    rows = []
    for key in sorted(results.keys()):
        avg = results[key]
        for cat_idx, cat_name in enumerate(categories):
            rows.append([key, cat_name, float(avg[cat_idx])])
    bar_table = wandb.Table(data=rows, columns=["layer_head", "category", "attention_fraction"])
    wandb.log({
        "attn_distribution": wandb.plot.bar(
            bar_table, "layer_head", "attention_fraction",
            title="Attention Distribution by Layer/Head"),
    })

    detail_rows = []
    for key in sorted(results.keys()):
        avg = results[key]
        detail_rows.append([key, float(avg[0]), float(avg[1]), float(avg[2]), float(avg[3])])
    detail_table = wandb.Table(
        data=detail_rows,
        columns=["layer_head", "parent", "prior_output", "prior_parent", "other"])
    wandb.log({"attn_distribution_table": detail_table})

    print(f"  Attn distribution: analyzed {total_cells} cells from {len(subset)} samples")


def get_parent_pattern(cell_values, pos, cell_width, stride):
    """Get the 3-bit parent pattern (0-7) for a given position."""
    row = pos // stride
    col = pos % stride
    actual_col = (col + 1) % cell_width
    nb_row = row - 1
    if nb_row < 0:
        return -1
    left = (actual_col - 1) % cell_width
    center = actual_col
    right = (actual_col + 1) % cell_width
    l_val = int(cell_values[nb_row * stride + left])
    c_val = int(cell_values[nb_row * stride + center])
    r_val = int(cell_values[nb_row * stride + right])
    return (l_val << 2) | (c_val << 1) | r_val


def extract_layer_outputs(model, X, device):
    """Extract hidden states after each layer, plus intermediate (after attn, before FFN).
    
    Returns dict with keys:
        'embedding', 'L0_post_attn', 'L0_post_ffn', 'L1_post_attn', 'L1_post_ffn', ...
    """
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        B, T = X.shape
        x = model.embedding(X)
        positions = torch.arange(T, device=X.device)
        pos_emb = model.positional_embedding(positions).unsqueeze(0)
        x = x + pos_emb
        mask = model.causal_mask[:T, :T]

        results = {'embedding': x.cpu().numpy()}
        for li, layer in enumerate(model.transformer_layers):
            # Pre-norm attention
            normed = layer.norm1(x)
            attn_out, _ = layer.self_attn(normed, normed, normed, attn_mask=mask, need_weights=False)
            x_post_attn = x + layer.dropout1(attn_out)
            results[f'L{li}_post_attn'] = x_post_attn.cpu().numpy()

            # FFN
            if layer.use_mlp:
                y = layer.norm2(x_post_attn)
                y = layer.linear2(layer.dropout(layer.activation(layer.linear1(y))))
                x = x_post_attn + layer.dropout2(y)
            else:
                x = x_post_attn
            results[f'L{li}_post_ffn'] = x.cpu().numpy()

    return results


def run_pattern_clustering(model, test_ds, device, cell_width, num_context_rows,
                           num_samples=200, seed=42):
    """PCA/t-SNE of hidden states colored by parent pattern, logged to wandb."""
    np.random.seed(seed)
    stride = cell_width + 1
    num_layers = len(list(model.transformer_layers))

    indices = np.random.choice(len(test_ds), min(num_samples, len(test_ds)), replace=False)
    subset = Subset(test_ds, indices)

    # Build stage names: embedding, L0_post_attn, L0_post_ffn, L1_post_attn, L1_post_ffn, ...
    stage_keys = ['embedding']
    for li in range(num_layers):
        stage_keys.append(f'L{li}_post_attn')
        stage_keys.append(f'L{li}_post_ffn')

    stage_hiddens = {k: [] for k in stage_keys}
    all_patterns = []
    all_out_vals = []

    loader = DataLoader(subset, batch_size=min(50, len(subset)), shuffle=False)
    for batch_X, batch_y, batch_mask in loader:
        outputs = extract_layer_outputs(model, batch_X, device)
        for b in range(batch_X.shape[0]):
            cell_values = batch_X[b].numpy()
            target_values = batch_y[b].numpy()
            T = len(cell_values)
            for pos in range(T):
                if batch_mask[b, pos].item() == 0:
                    continue
                if pos % stride == cell_width:
                    continue
                row = pos // stride
                if row < num_context_rows:
                    continue
                pattern = get_parent_pattern(cell_values, pos, cell_width, stride)
                if pattern < 0:
                    continue
                out_val = int(target_values[pos])
                all_patterns.append(pattern)
                all_out_vals.append(out_val)
                for k in stage_keys:
                    stage_hiddens[k].append(outputs[k][b, pos])

    for k in stage_keys:
        stage_hiddens[k] = np.array(stage_hiddens[k])
    all_patterns = np.array(all_patterns)
    all_out_vals = np.array(all_out_vals)

    total_cells = len(all_patterns)
    print(f"  Pattern clustering: {total_cells} cells from {len(subset)} samples")

    # === Color schemes ===
    base_colors_rgb = plt.cm.Set1(np.linspace(0, 1, 8))[:, :3]
    pattern_names = [f"{p:03b}" for p in range(8)]

    # Pretty stage names for titles
    stage_titles = {'embedding': 'Embedding'}
    for li in range(num_layers):
        stage_titles[f'L{li}_post_attn'] = f'L{li} Post-Attn'
        stage_titles[f'L{li}_post_ffn'] = f'L{li} Post-FFN'

    # ============ PCA by pattern ============
    n_stages = len(stage_keys)
    fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 5))
    if n_stages == 1:
        axes = [axes]

    for si, key in enumerate(stage_keys):
        pca = PCA(n_components=2)
        coords = pca.fit_transform(stage_hiddens[key])
        for p in range(8):
            mask_p = all_patterns == p
            axes[si].scatter(coords[mask_p, 0], coords[mask_p, 1],
                             c=[base_colors_rgb[p]], label=pattern_names[p] if si == 0 else "",
                             alpha=0.4, s=8)
        v1 = pca.explained_variance_ratio_[0]
        v2 = pca.explained_variance_ratio_[1]
        axes[si].set_title(f"{stage_titles[key]}\n({v1:.1%}, {v2:.1%})", fontsize=10)
        axes[si].set_xlabel("PC1")
        axes[si].set_ylabel("PC2")

    axes[0].legend(markerscale=3, fontsize=7)
    fig.suptitle("PCA by Parent Pattern", fontsize=14)
    fig.tight_layout()
    wandb.log({"pattern_clustering_pca_by_pattern": wandb.Image(fig)})
    plt.close(fig)

    # ============ PCA by output value ============
    fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 5))
    if n_stages == 1:
        axes = [axes]

    for si, key in enumerate(stage_keys):
        pca = PCA(n_components=2)
        coords = pca.fit_transform(stage_hiddens[key])
        for v, c in [(0, 'blue'), (1, 'red')]:
            mask_v = all_out_vals == v
            axes[si].scatter(coords[mask_v, 0], coords[mask_v, 1],
                             c=c, label=f"out={v}" if si == 0 else "", alpha=0.3, s=8)
        v1 = pca.explained_variance_ratio_[0]
        v2 = pca.explained_variance_ratio_[1]
        axes[si].set_title(f"{stage_titles[key]}\n({v1:.1%}, {v2:.1%})", fontsize=10)
        axes[si].set_xlabel("PC1")
        axes[si].set_ylabel("PC2")

    axes[0].legend(markerscale=3, fontsize=10)
    fig.suptitle("PCA by Output Value (0 vs 1)", fontsize=14)
    fig.tight_layout()
    wandb.log({"pattern_clustering_pca_by_output": wandb.Image(fig)})
    plt.close(fig)

    # ============ t-SNE by pattern (all stages side by side) ============
    max_tsne = 5000
    if len(all_patterns) > max_tsne:
        tsne_idx = np.random.choice(len(all_patterns), max_tsne, replace=False)
    else:
        tsne_idx = np.arange(len(all_patterns))
    tsne_patterns = all_patterns[tsne_idx]
    tsne_out_vals = all_out_vals[tsne_idx]

    fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 5))
    if n_stages == 1:
        axes = [axes]

    for si, key in enumerate(stage_keys):
        hiddens = stage_hiddens[key][tsne_idx]
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
        coords = tsne.fit_transform(hiddens)
        for p in range(8):
            mask_p = tsne_patterns == p
            axes[si].scatter(coords[mask_p, 0], coords[mask_p, 1],
                             c=[base_colors_rgb[p]], label=pattern_names[p] if si == 0 else "",
                             alpha=0.5, s=8)
        axes[si].set_title(f"{stage_titles[key]}", fontsize=10)

    axes[0].legend(markerscale=3, fontsize=7)
    fig.suptitle("t-SNE by Parent Pattern", fontsize=14)
    fig.tight_layout()
    wandb.log({"pattern_clustering_tsne_by_pattern": wandb.Image(fig)})
    plt.close(fig)

    # ============ t-SNE by output (all stages side by side) ============
    fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 5))
    if n_stages == 1:
        axes = [axes]

    for si, key in enumerate(stage_keys):
        hiddens = stage_hiddens[key][tsne_idx]
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
        coords = tsne.fit_transform(hiddens)
        for v, c in [(0, 'blue'), (1, 'red')]:
            mask_v = tsne_out_vals == v
            axes[si].scatter(coords[mask_v, 0], coords[mask_v, 1],
                             c=c, label=f"out={v}" if si == 0 else "", alpha=0.5, s=8)
        axes[si].set_title(f"{stage_titles[key]}", fontsize=10)

    axes[0].legend(markerscale=3, fontsize=10)
    fig.suptitle("t-SNE by Output Value (0 vs 1)", fontsize=14)
    fig.tight_layout()
    wandb.log({"pattern_clustering_tsne_by_output": wandb.Image(fig)})
    plt.close(fig)

    print("  Pattern clustering: logged to wandb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--cell_width", type=int, required=True)
    parser.add_argument("--show_all_errors", action="store_true",
                        help="Show all errors (default: cap at --max_errors)")
    parser.add_argument("--max_errors", type=int, default=50)
    parser.add_argument("--success_per_row", type=int, default=2,
                        help="Number of success cases to sample per row")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--project_name", type=str, default="eca_transformer")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default="menghan-xu-cornell-university")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_dist_samples", type=int, default=200,
                        help="Number of samples for attention distribution analysis")
    parser.add_argument("--num_context_rows", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    stride = args.cell_width + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    model, cfg = load_model(args.ckpt, device)
    print(f"Model config: {cfg}")

    test_ds = ECADataset(args.test_path)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    rules = test_ds.rules

    print("Scanning for errors and successes...")
    errors, successes = scan_samples(
        model, test_loader, device, args.cell_width,
        show_all_errors=args.show_all_errors,
        max_errors=args.max_errors,
        success_per_row=args.success_per_row,
    )
    print(f"Found {len(errors)} error samples, {len(successes)} success cases")

    # Init wandb
    run_name = args.run_name or f"analysis_{args.ckpt.split('/')[-2]}"
    wandb.init(entity=args.wandb_entity, project=args.project_name,
               name=run_name, job_type="analysis")
    wandb.config.update({"ckpt": args.ckpt, "cell_width": args.cell_width, **cfg})

    columns = ["type", "sample_idx", "rule", "query_row", "query_col",
               "pred", "truth", "pattern", "num_errors", "prior_count", "attention_vis"]
    table = wandb.Table(columns=columns)

    error_prior_counts = []
    success_prior_counts = []

    def add_case(X, y, q_pos, global_idx, is_correct, num_errs):
        attn_maps, preds = extract_attention_maps(model, X.unsqueeze(0), device)
        rule_num = int(rules[global_idx]) if rules is not None else -1
        pred_val = preds[q_pos]
        true_val = y[q_pos].item()
        q_row = q_pos // stride
        q_col = q_pos % stride
        actual_col = (q_col + 1) % args.cell_width

        prior_count = count_prior_patterns(X.numpy(), q_pos, args.cell_width, stride)
        if is_correct:
            success_prior_counts.append(prior_count)
        else:
            error_prior_counts.append(prior_count)

        nb_row = q_row - 1
        if nb_row >= 0:
            pattern = ""
            for dc in [-1, 0, 1]:
                c = (actual_col + dc) % args.cell_width
                idx = nb_row * stride + c
                pattern += str(int(X[idx].item()))
            pattern += f"->{int(true_val)}"
        else:
            pattern = "first_row"

        html = generate_attention_html(
            cell_values=X.numpy(),
            attn_layers=attn_maps,
            query_flat_idx=q_pos,
            cell_width=args.cell_width,
            stride=stride,
            pred_val=pred_val,
            true_val=true_val,
            is_correct=is_correct,
        )
        case_type = "success" if is_correct else "error"
        table.add_data(case_type, global_idx, rule_num, q_row, actual_col,
                       int(pred_val), int(true_val), pattern, num_errs, prior_count,
                       wandb.Html(html))

    # Log all error cases
    for X, y, err_positions, gidx in errors:
        for q_pos in err_positions:
            add_case(X, y, q_pos, gidx, is_correct=False, num_errs=len(err_positions))

    # Log success cases
    for X, y, pos, gidx in successes:
        add_case(X, y, pos, gidx, is_correct=True, num_errs=0)

    wandb.log({"error_analysis": table})

    # Stats
    avg_error = np.mean(error_prior_counts) if error_prior_counts else 0
    avg_success = np.mean(success_prior_counts) if success_prior_counts else 0

    # Bar chart
    bar_data = [[label, val] for label, val in [("error", avg_error), ("success", avg_success)]]
    bar_table = wandb.Table(data=bar_data, columns=["type", "avg_prior_count"])
    wandb.log({
        "prior_count_comparison": wandb.plot.bar(
            bar_table, "type", "avg_prior_count",
            title="Avg Prior Pattern Occurrences: Error vs Success"),
        "avg_prior_count_error": avg_error,
        "avg_prior_count_success": avg_success,
    })

    total_error_cells = sum(len(e[2]) for e in errors)
    wandb.log({
        "total_samples_with_errors": len(errors),
        "total_error_cells": total_error_cells,
        "total_success_cases_logged": len(successes),
    })

    print(f"Logged {total_error_cells} error cells + {len(successes)} success cases")

    # Attention distribution analysis
    print("Running attention distribution analysis...")
    run_attn_distribution(model, test_ds, device, args.cell_width,
                          args.attn_dist_samples, args.num_context_rows, args.seed)

    # Pattern clustering analysis
    print("Running pattern clustering analysis...")
    run_pattern_clustering(model, test_ds, device, args.cell_width,
                           args.num_context_rows, num_samples=args.attn_dist_samples,
                           seed=args.seed)

    wandb.finish()


if __name__ == "__main__":
    main()