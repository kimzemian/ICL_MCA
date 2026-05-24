"""
Attention ablation study.

Usage:
    # Full ablation (all layers + all heads)
    python attn_ablation.py --ckpt /path/to/best.pt --test_path /path/to/test.npz --cell_width 15

    # Only specific layers
    python attn_ablation.py ... --layers 1 2

    # Only per-head ablation
    python attn_ablation.py ... --heads_only

    # Only layer-level ablation (skip per-head)
    python attn_ablation.py ... --no_per_head
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from data_generate import ECADataset
from torch.utils.data import DataLoader
from models import VSimpleTransformer
from eval import evaluate


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


def _freeze_training(layer):
    original = {
        'training': layer.training,
        'dropout1_p': layer.dropout1.p,
    }
    layer.training = True
    layer.train = lambda mode=True: layer
    layer.dropout1.p = 0.0
    return original


def _restore_training(layer, original):
    layer.training = original['training']
    if 'train' in layer.__dict__:
        del layer.__dict__['train']
    layer.dropout1.p = original['dropout1_p']


def patch_uniform_attention(model, layer_indices):
    saved = {}
    for i in layer_indices:
        layer = model.transformer_layers[i]
        saved[i] = {'sa_block': layer._sa_block, **_freeze_training(layer)}

        def make_uniform_sa(layer_obj):
            def uniform_sa_block(x, attn_mask=None, key_padding_mask=None, is_causal=False, **kwargs):
                B, T, D = x.shape
                causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                attend_count = (~causal).float().sum(dim=-1, keepdim=True)
                uniform_weights = (~causal).float() / attend_count

                sa = layer_obj.self_attn
                qkv = nn.functional.linear(x, sa.in_proj_weight, sa.in_proj_bias)
                _, _, v = qkv.chunk(3, dim=-1)
                out = torch.matmul(uniform_weights.unsqueeze(0), v)
                out = nn.functional.linear(out, sa.out_proj.weight, sa.out_proj.bias)
                return out
            return uniform_sa_block

        layer._sa_block = make_uniform_sa(layer)

    def restore():
        for i in layer_indices:
            layer = model.transformer_layers[i]
            layer._sa_block = saved[i]['sa_block']
            _restore_training(layer, saved[i])
    return restore


def patch_zero_attention(model, layer_indices):
    saved = {}
    for i in layer_indices:
        layer = model.transformer_layers[i]
        saved[i] = {'sa_block': layer._sa_block, **_freeze_training(layer)}

        def make_zero_sa():
            def zero_sa_block(x, *args, **kwargs):
                return torch.zeros_like(x)
            return zero_sa_block

        layer._sa_block = make_zero_sa()

    def restore():
        for i in layer_indices:
            layer = model.transformer_layers[i]
            layer._sa_block = saved[i]['sa_block']
            _restore_training(layer, saved[i])
    return restore


def patch_head_zero(model, layer_idx, head_indices):
    layer = model.transformer_layers[layer_idx]
    saved = {'sa_block': layer._sa_block, **_freeze_training(layer)}

    def make_head_zero_sa(layer_obj, zero_heads):
        def head_zero_sa_block(x, attn_mask=None, key_padding_mask=None, is_causal=False, **kwargs):
            sa = layer_obj.self_attn
            num_heads = sa.num_heads
            d_head = sa.embed_dim // num_heads
            B, T, D = x.shape

            qkv = nn.functional.linear(x, sa.in_proj_weight, sa.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, num_heads, d_head).transpose(1, 2)
            k = k.view(B, T, num_heads, d_head).transpose(1, 2)
            v = v.view(B, T, num_heads, d_head).transpose(1, 2)

            scale = d_head ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=-1)

            head_out = torch.matmul(attn_weights, v)
            for h in zero_heads:
                head_out[:, h, :, :] = 0.0

            out = head_out.transpose(1, 2).contiguous().view(B, T, D)
            out = nn.functional.linear(out, sa.out_proj.weight, sa.out_proj.bias)
            return out
        return head_zero_sa_block

    layer._sa_block = make_head_zero_sa(layer, head_indices)

    def restore():
        layer._sa_block = saved['sa_block']
        _restore_training(layer, saved)
    return restore


def patch_head_uniform(model, layer_idx, head_indices):
    layer = model.transformer_layers[layer_idx]
    saved = {'sa_block': layer._sa_block, **_freeze_training(layer)}

    def make_head_uniform_sa(layer_obj, uniform_heads):
        def head_uniform_sa_block(x, attn_mask=None, key_padding_mask=None, is_causal=False, **kwargs):
            sa = layer_obj.self_attn
            num_heads = sa.num_heads
            d_head = sa.embed_dim // num_heads
            B, T, D = x.shape

            qkv = nn.functional.linear(x, sa.in_proj_weight, sa.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, num_heads, d_head).transpose(1, 2)
            k = k.view(B, T, num_heads, d_head).transpose(1, 2)
            v = v.view(B, T, num_heads, d_head).transpose(1, 2)

            scale = d_head ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=-1)

            attend_count = (~causal).float().sum(dim=-1, keepdim=True)
            uniform_w = (~causal).float() / attend_count
            for h in uniform_heads:
                attn_weights[:, h, :, :] = uniform_w.unsqueeze(0)

            head_out = torch.matmul(attn_weights, v)
            out = head_out.transpose(1, 2).contiguous().view(B, T, D)
            out = nn.functional.linear(out, sa.out_proj.weight, sa.out_proj.bias)
            return out
        return head_uniform_sa_block

    layer._sa_block = make_head_uniform_sa(layer, head_indices)

    def restore():
        layer._sa_block = saved['sa_block']
        _restore_training(layer, saved)
    return restore


def run_eval(model, test_loader, criterion, device, cell_width, rules):
    metrics = evaluate(model, test_loader, criterion, device,
                       cell_width=cell_width, rules=rules)
    return metrics["cell_acc"], metrics.get("zero_error_rate", -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--cell_width", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--layers", type=int, nargs='+', default=None,
                        help="Which layers to ablate (default: all)")
    parser.add_argument("--no_per_head", action="store_true",
                        help="Skip per-head ablation")
    parser.add_argument("--heads_only", action="store_true",
                        help="Skip layer-level, only do per-head ablation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    model, cfg = load_model(args.ckpt, device)
    num_layers = len(cfg["heads_list"])
    print(f"Model: {num_layers} layers, heads={cfg['heads_list']}, d={cfg['hidden_size']}")

    target_layers = args.layers if args.layers is not None else list(range(num_layers))
    print(f"Target layers: {target_layers}")

    test_ds = ECADataset(args.test_path)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    rules = test_ds.rules

    # Baseline
    print("\n=== Baseline (no ablation) ===")
    baseline_acc, baseline_zer = run_eval(model, test_loader, criterion, device, args.cell_width, rules)
    print(f"  Cell acc: {baseline_acc:.6f}  Zero-error: {baseline_zer:.6f}")

    # === Layer-level ablation ===
    if not args.heads_only:
        for mode_name, patch_fn in [("UNIFORM", patch_uniform_attention), ("ZERO", patch_zero_attention)]:
            print(f"\n=== Single layer {mode_name} ablation ===")
            print(f"{'Ablated':<12} {'Cell Acc':<12} {'Zero-Error':<12} {'Δ ZER'}")
            print(f"{'-'*48}")

            for layer_idx in target_layers:
                restore = patch_fn(model, [layer_idx])
                acc, zer = run_eval(model, test_loader, criterion, device, args.cell_width, rules)
                restore()
                print(f"Layer {layer_idx:<6} {acc:<12.6f} {zer:<12.6f} {zer - baseline_zer:+.6f}")

        # All target layers zero
        print(f"\n=== All target layers ZERO ablation ===")
        restore = patch_zero_attention(model, target_layers)
        acc, zer = run_eval(model, test_loader, criterion, device, args.cell_width, rules)
        restore()
        print(f"Layers {target_layers}  {acc:<12.6f} {zer:<12.6f} {zer - baseline_zer:+.6f}")

    # === Per-head ablation ===
    if not args.no_per_head:
        for mode_name, patch_fn in [("ZERO", patch_head_zero), ("UNIFORM", patch_head_uniform)]:
            for layer_idx in target_layers:
                num_heads = cfg["heads_list"][layer_idx]
                print(f"\n=== Layer {layer_idx} per-head {mode_name} ablation ({num_heads} heads) ===")
                print(f"{'Head':<12} {'Cell Acc':<12} {'Zero-Error':<12} {'Δ ZER'}")
                print(f"{'-'*48}")

                for h in range(num_heads):
                    restore = patch_fn(model, layer_idx, [h])
                    acc, zer = run_eval(model, test_loader, criterion, device, args.cell_width, rules)
                    restore()
                    print(f"H{h:<10} {acc:<12.6f} {zer:<12.6f} {zer - baseline_zer:+.6f}")


if __name__ == "__main__":
    main()