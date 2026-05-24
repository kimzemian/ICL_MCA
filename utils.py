import torch
import numpy as np
import wandb


def extract_attention_maps(model, x, device):
    model.eval()
    attn_maps = {}
    hooks = []

    for i, layer in enumerate(model.transformer_layers):
        orig_forward = layer.self_attn.forward

        def make_hook(layer_idx, orig_fn):
            def hooked_forward(*args, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False
                out, weights = orig_fn(*args, **kwargs)
                attn_maps[layer_idx] = weights.detach().cpu().numpy()
                return out, weights
            return hooked_forward

        layer.self_attn.forward = make_hook(i, orig_forward)
        hooks.append((layer, orig_forward))

    with torch.no_grad():
        x = x.to(device)
        output = model(x)

    for layer, orig_forward in hooks:
        layer.self_attn.forward = orig_forward

    for k in attn_maps:
        attn_maps[k] = attn_maps[k][0]

    preds = output[0].argmax(dim=-1).cpu().numpy()
    return attn_maps, preds


def count_prior_patterns(cell_values, query_flat_idx, cell_width, stride):
    """Count how many times the query's parent pattern appeared in earlier rows."""
    query_row = query_flat_idx // stride
    query_col = query_flat_idx % stride
    actual_col = query_col + 1
    nb_row = query_row - 1

    if nb_row < 0:
        return 0

    nb_cols = [(actual_col - 1) % cell_width, actual_col % cell_width, (actual_col + 1) % cell_width]
    pattern_bits = []
    for c in nb_cols:
        pattern_bits.append(int(cell_values[nb_row * stride + c]))
    query_pattern = tuple(pattern_bits)

    count = 0
    for r in range(0, query_row - 1):
        for c in range(cell_width):
            left = (c - 1) % cell_width
            right = (c + 1) % cell_width
            bits = (int(cell_values[r * stride + left]),
                    int(cell_values[r * stride + c]),
                    int(cell_values[r * stride + right]))
            if bits == query_pattern:
                count += 1
    return count


def generate_attention_html(cell_values, attn_layers, query_flat_idx, cell_width, stride,
                            pred_val, true_val, vis_context_rows=100, is_correct=False):
    query_row = query_flat_idx // stride
    query_col = query_flat_idx % stride
    # target[q_pos] predicts the NEXT token, i.e. cell at actual_col
    actual_col = (query_col + 1) % cell_width

    row_start = max(0, query_row - vis_context_rows)
    row_end = min(len(cell_values) // stride, query_row + 2)

    nb_row = query_row - 1
    nb_cols = [(actual_col - 1) % cell_width, actual_col % cell_width, (actual_col + 1) % cell_width]

    # Compute query's parent pattern
    query_pattern = None
    if nb_row >= 0:
        pattern_vals = []
        pattern_bits = []
        for c in nb_cols:
            idx = nb_row * stride + c
            val = int(cell_values[idx])
            pattern_vals.append(str(val))
            pattern_bits.append(val)
        pattern_str = "".join(pattern_vals) + "->" + str(int(true_val))
        query_pattern = tuple(pattern_bits)
    else:
        pattern_str = "?"

    # Scan rows 0 to query_row-2 for same pattern
    pattern_match_cells = set()   # (row, col) of parent cells (green border)
    pattern_output_cells = set()  # (row, col) of output cells (green bg)
    match_count = 0
    if query_pattern is not None:
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
                    match_count += 1
                    pattern_match_cells.add((r, left))
                    pattern_match_cells.add((r, c))
                    pattern_match_cells.add((r, right))
                    pattern_output_cells.add((r + 1, c))

    layer_keys = sorted(attn_layers.keys())
    result_color = "green" if is_correct else "red"
    result_label = "CORRECT" if is_correct else "ERROR"

    html_parts = []
    html_parts.append(f"""
<div style="font-family:system-ui,sans-serif;font-size:13px;max-width:800px;">
<div style="margin-bottom:10px;">
  <b style="color:{result_color}">{result_label}</b> &nbsp;|&nbsp;
  <b>Query:</b> row {query_row}, col {actual_col} &nbsp;|&nbsp;
  <b>Pred:</b> {int(pred_val)} &nbsp;
  <b>Truth:</b> {int(true_val)} &nbsp;|&nbsp;
  <b>Pattern:</b> {pattern_str} &nbsp;|&nbsp;
  <b>Prior occurrences:</b> {match_count}
</div>
""")

    for li in layer_keys:
        attn = attn_layers[li]
        num_heads = attn.shape[0]
        query_attn = attn[:, query_flat_idx, :]

        html_parts.append(f'<div style="margin-bottom:14px;"><b>Layer {li}</b> ({num_heads} heads)</div>')
        html_parts.append('<div style="display:flex;gap:16px;flex-wrap:wrap;">')

        for h in range(num_heads):
            h_attn = query_attn[h]
            visible_indices = []
            for r in range(row_start, row_end):
                for c in range(cell_width):
                    idx = r * stride + c
                    if idx < query_flat_idx and idx < len(cell_values):
                        visible_indices.append(idx)

            max_a = max([h_attn[i] for i in visible_indices]) if visible_indices else 1e-8

            html_parts.append('<div style="background:#f8f8f8;border-radius:6px;padding:8px;overflow-x:auto;">')
            html_parts.append(f'<div style="font-size:11px;font-weight:600;margin-bottom:4px;">Head {h}</div>')

            for r in range(row_start, row_end):
                html_parts.append('<div style="display:flex;gap:1px;margin-bottom:1px;white-space:nowrap;">')
                html_parts.append(f'<div style="width:20px;font-size:9px;color:#999;text-align:right;padding-top:2px;margin-right:2px;flex-shrink:0;">t{r}</div>')

                for c in range(cell_width):
                    flat = r * stride + c
                    is_query = (r == query_row and c == actual_col)
                    is_future = flat > query_flat_idx
                    is_nb = (r == nb_row and c in nb_cols and 0 <= c < cell_width)
                    is_pattern_parent = (r, c) in pattern_match_cells
                    is_pattern_output = (r, c) in pattern_output_cells

                    val = cell_values[flat] if flat < len(cell_values) else 0
                    a = h_attn[flat] if flat < len(h_attn) else 0
                    t = min(a / max_a, 1.0) if max_a > 0 else 0

                    if is_query:
                        bg, border, txt_color, display_val = "#dbeafe", "2px solid #3b82f6", "#1e40af", "?"
                    elif is_future:
                        bg, border, txt_color, display_val = "#f5f5f5", "1px solid transparent", "#ccc", str(int(val))
                    else:
                        r_c = int(247 - t * 215)
                        g_c = int(249 - t * 220)
                        b_c = int(253 - t * 120)
                        bg = f"rgb({r_c},{g_c},{b_c})"
                        txt_color = "#fff" if t > 0.5 else "#333"
                        border = "1px solid transparent"
                        display_val = str(int(val))

                    if is_nb:
                        border = "2px solid #e24b4a"
                    elif is_pattern_output:
                        border = "2px solid #16a34a"
                    elif is_pattern_parent:
                        border = "2px solid #86efac"

                    title_str = f"attn={a:.4f}" if not is_query and not is_future else ""
                    html_parts.append(
                        f'<div title="{title_str}" style="'
                        f'min-width:14px;height:16px;display:flex;align-items:center;justify-content:center;'
                        f'font-size:7px;font-weight:600;border-radius:1px;'
                        f'background:{bg};color:{txt_color};border:{border};'
                        f'">{display_val}</div>'
                    )

                html_parts.append('</div>')

            top_k = min(5, len(visible_indices))
            if visible_indices:
                sorted_idx = sorted(visible_indices, key=lambda i: h_attn[i], reverse=True)[:top_k]
                top_strs = [f"(t{si // stride},c{si % stride})={h_attn[si]:.3f}" for si in sorted_idx]
                html_parts.append(f'<div style="font-size:9px;color:#888;margin-top:4px;">Top: {", ".join(top_strs)}</div>')

            html_parts.append('</div>')

        html_parts.append('</div>')

    html_parts.append("""
<div style="display:flex;gap:12px;margin-top:12px;font-size:10px;color:#888;flex-wrap:wrap;">
  <span>Red border = query neighbors</span>
  <span>Blue = query cell</span>
  <span>Dark green border = prior pattern output</span>
  <span>Light green border = prior pattern parents</span>
  <span>Darker blue = higher attention</span>
</div>
""")
    html_parts.append('</div>')
    return "".join(html_parts)


def log_sample_analysis(model, loader, device, cell_width, epoch, rules=None,
                        max_error_cases=15, max_success_cases=15, max_samples_scan=500):
    stride = cell_width + 1
    model.eval()

    error_samples = []
    success_samples = []

    sample_offset = 0
    with torch.no_grad():
        for X, y, mask in loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            output = model(X)
            preds = output.argmax(dim=-1)

            T = X.shape[1]
            eval_mask = mask > 0
            is_error = (preds != y) & eval_mask
            has_error = is_error.any(dim=1)
            has_eval = eval_mask.any(dim=1)

            for b in range(X.shape[0]):
                global_idx = sample_offset + b
                if not has_eval[b]:
                    continue
                if has_error[b] and len(error_samples) < max_error_cases:
                    err_pos = is_error[b].nonzero(as_tuple=True)[0].cpu().tolist()
                    error_samples.append((X[b].cpu(), y[b].cpu(), err_pos, global_idx))
                elif (not has_error[b]) and len(success_samples) < max_success_cases:
                    ok_pos = eval_mask[b].nonzero(as_tuple=True)[0].cpu().tolist()
                    success_samples.append((X[b].cpu(), y[b].cpu(), ok_pos, global_idx))

            sample_offset += X.shape[0]
            if (len(error_samples) >= max_error_cases
                and len(success_samples) >= max_success_cases):
                break
            if sample_offset >= max_samples_scan:
                break

    columns = ["type", "sample_idx", "rule", "query_row", "query_col",
               "pred", "truth", "pattern", "num_errors", "attention_vis"]
    table = wandb.Table(columns=columns)

    def add_case(X, y, q_pos, global_idx, is_correct, num_errs):
        attn_maps, preds = extract_attention_maps(model, X.unsqueeze(0), device)
        rule_num = int(rules[global_idx]) if rules is not None else -1
        pred_val = preds[q_pos]
        true_val = y[q_pos].item()
        q_row = q_pos // stride
        q_col = q_pos % stride
        actual_col = (q_col + 1) % cell_width

        nb_row = q_row - 1
        if nb_row >= 0:
            pattern = ""
            for dc in [-1, 0, 1]:
                c = (actual_col + dc) % cell_width
                idx = nb_row * stride + c
                pattern += str(int(X[idx].item()))
            pattern += f"->{int(true_val)}"
        else:
            pattern = "first_row"

        html = generate_attention_html(
            cell_values=X.numpy(),
            attn_layers=attn_maps,
            query_flat_idx=q_pos,
            cell_width=cell_width,
            stride=stride,
            pred_val=pred_val,
            true_val=true_val,
            is_correct=is_correct,
        )
        case_type = "success" if is_correct else "error"
        table.add_data(case_type, global_idx, rule_num, q_row, actual_col,
                       int(pred_val), int(true_val), pattern, num_errs, wandb.Html(html))

    for X, y, err_positions, gidx in error_samples:
        add_case(X, y, err_positions[0], gidx, is_correct=False, num_errs=len(err_positions))

    for X, y, ok_positions, gidx in success_samples:
        q_pos = ok_positions[len(ok_positions) // 2]
        add_case(X, y, q_pos, gidx, is_correct=True, num_errs=0)

    wandb.log({f"sample_analysis_ep{epoch+1}": table})
    print(f"  [Epoch {epoch+1}] Logged {len(error_samples)} error + {len(success_samples)} success cases")