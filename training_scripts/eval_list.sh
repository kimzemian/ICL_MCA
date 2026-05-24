#!/bin/bash
CKPT_DIR=/share/dean/mx253/icl_ca/checkpoint
RESULTS_FILE=/tmp/eval_results.txt

> $RESULTS_FILE

# === L16 T10 M4 (original) ===
TEST_PATH=/share/dean/mx253/icl_ca/eca_data/L16_T10_M4_seed42_30k/mixed_test.npz
CELL_WIDTH=15
RUNS_L16=(
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-2-2-d128-lr1e-3-wd0.2
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-2-4-d128-lr1e-3-wd0.2
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-4-4-d128-lr1e-3-wd0.2
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-8-8-d128-lr1e-3-wd0.2
    ca-cmixed-L16_T10_M4_seed42_30k-layers4-heads-2-2-2-2-d128-lr1e-3-wd0.2
    ca-cmixed-L16_T10_M4_seed42_30k-layers4-heads-2-4-4-2-d128-lr1e-3-wd0.2
    ca-cmixed-L16_T10_M4_seed42_30k-layers4-heads-4-4-4-4-d128-lr1e-3-wd0.2
    ca-cmixed-L16_T10_M4_seed42_30k-layers6-heads-2-2-2-2-2-2-d128-lr1e-3-wd0.2
    ca-cmixed-L16_T10_M4_seed42_30k-layers4-heads-2-4-4-2-d256-lr1e-3-wd0.05-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers4-heads-2-4-4-2-d256-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers4-heads-2-4-4-2-d128-lr1e-3-wd0.0-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers4-heads-2-4-4-2-d128-lr1e-3-wd0.05-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-4-4-d256-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-4-1-d256-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers3-heads-4-4-2-d256-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers3-heads-2-4-4-d256-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-3-3-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-3-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers1-heads-4-d256-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-d256-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-2-1-d256-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-d256-lr1e-3-wd0.0-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-d256-lr1e-3-wd0.2-ep1000
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-d512-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-d512-lr1e-3-wd0.0-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-0-d512-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp0-1-d512-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp0-0-d512-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-3-1-mlp1-0-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-3-1-mlp1-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-3-1-mlp0-0-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-3-1-mlp0-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-4-1-mlp0-0-d512-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-4-1-mlp0-1-d512-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-4-1-mlp1-0-d512-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-4-1-mlp1-1-d512-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers1-heads-8-mlp1-1-d512-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-2-1-mlp1-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-2-1-mlp1-0-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-2-1-mlp0-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-2-1-mlp0-0-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-3-1-mlp1-1-d96-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d64-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d8-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-3-1-mlp1-1-d12-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-3-1-mlp1-1-d24-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d512-lr1e-3-wd0.2-ep300-sanity
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d128-dmlp8-lr1e-3-wd0.2-ep300
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d128-dmlp16-lr1e-3-wd0.2-ep300
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d128-dmlp32-lr1e-3-wd0.2-ep300
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d128-dmlp64-lr1e-3-wd0.2-ep300
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d128-dmlp128-lr1e-3-wd0.2-ep300
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d64-dmlp128-lr1e-3-wd0.2-ep300
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d32-dmlp128-lr1e-3-wd0.2-ep300
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d16-dmlp32-lr1e-3-wd0.2-ep300
    ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-mlp1-1-d15-dmlp32-lr1e-3-wd0.2-ep300
)
for run_name in "${RUNS_L16[@]}"; do
    best_ckpt="${CKPT_DIR}/${run_name}/${run_name}_best.pt"
    [ ! -f "$best_ckpt" ] && echo "SKIP: $run_name" && continue
    # Parse dmlp{N} from run_name; old ckpts didn't save ffn_dim_list in cfg, so we override.
    dmlp_val=$(echo "$run_name" | grep -oP 'dmlp\K[0-9]+' || true)
    nlayers=$(echo "$run_name" | grep -oP 'layers\K[0-9]+')
    extra_args=""
    if [ -n "$dmlp_val" ]; then
        extra_args="--ffn_dim_list $(yes $dmlp_val | head -n $nlayers | tr '\n' ' ')"
    fi
    output=$(python eval.py --checkpoint_path "$best_ckpt" --test_path "$TEST_PATH" --cell_width $CELL_WIDTH --device cuda $extra_args 2>/dev/null)
    zer=$(echo "$output" | grep "Zero-error" | awk '{print $2}')
    acc=$(echo "$output" | grep "Cell acc" | awk '{print $3}')
    [ -n "$zer" ] && echo "$zer $acc $run_name" >> $RESULTS_FILE
done

# === L16 T16 M4 ===
TEST_PATH=/share/dean/mx253/icl_ca/eca_data/L16_T16_M4_seed42_120k/mixed_test.npz
CELL_WIDTH=15
RUNS=(
    ca-cmixed-L16_T16_M4_seed42_120k-layers2-heads-3-1-mlp1-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T16_M4_seed42_120k-layers2-heads-1-1-mlp1-1-d512-lr1e-3-wd0.2-ep500
)
for run_name in "${RUNS[@]}"; do
    best_ckpt="${CKPT_DIR}/${run_name}/${run_name}_best.pt"
    [ ! -f "$best_ckpt" ] && echo "SKIP: $run_name" && continue
    # Parse dmlp{N} from run_name; old ckpts didn't save ffn_dim_list in cfg, so we override.
    dmlp_val=$(echo "$run_name" | grep -oP 'dmlp\K[0-9]+' || true)
    nlayers=$(echo "$run_name" | grep -oP 'layers\K[0-9]+')
    extra_args=""
    if [ -n "$dmlp_val" ]; then
        extra_args="--ffn_dim_list $(yes $dmlp_val | head -n $nlayers | tr '\n' ' ')"
    fi
    output=$(python eval.py --checkpoint_path "$best_ckpt" --test_path "$TEST_PATH" --cell_width $CELL_WIDTH --device cuda $extra_args 2>/dev/null)
    zer=$(echo "$output" | grep "Zero-error" | awk '{print $2}')
    acc=$(echo "$output" | grep "Cell acc" | awk '{print $3}')
    [ -n "$zer" ] && echo "$zer $acc $run_name" >> $RESULTS_FILE
done

# === L16 T14 M4 ===
TEST_PATH=/share/dean/mx253/icl_ca/eca_data/L16_T14_M4_seed42_120k/mixed_test.npz
CELL_WIDTH=15
RUNS=(
    ca-cmixed-L16_T14_M4_seed42_120k-layers2-heads-3-1-mlp1-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T14_M4_seed42_120k-layers2-heads-1-1-mlp1-1-d512-lr1e-3-wd0.2-ep500
)
for run_name in "${RUNS[@]}"; do
    best_ckpt="${CKPT_DIR}/${run_name}/${run_name}_best.pt"
    [ ! -f "$best_ckpt" ] && echo "SKIP: $run_name" && continue
    # Parse dmlp{N} from run_name; old ckpts didn't save ffn_dim_list in cfg, so we override.
    dmlp_val=$(echo "$run_name" | grep -oP 'dmlp\K[0-9]+' || true)
    nlayers=$(echo "$run_name" | grep -oP 'layers\K[0-9]+')
    extra_args=""
    if [ -n "$dmlp_val" ]; then
        extra_args="--ffn_dim_list $(yes $dmlp_val | head -n $nlayers | tr '\n' ' ')"
    fi
    output=$(python eval.py --checkpoint_path "$best_ckpt" --test_path "$TEST_PATH" --cell_width $CELL_WIDTH --device cuda $extra_args 2>/dev/null)
    zer=$(echo "$output" | grep "Zero-error" | awk '{print $2}')
    acc=$(echo "$output" | grep "Cell acc" | awk '{print $3}')
    [ -n "$zer" ] && echo "$zer $acc $run_name" >> $RESULTS_FILE
done

# === L16 T12 M4 ===
TEST_PATH=/share/dean/mx253/icl_ca/eca_data/L16_T12_M4_seed42_120k/mixed_test.npz
CELL_WIDTH=15
RUNS=(
    ca-cmixed-L16_T12_M4_seed42_120k-layers2-heads-3-1-mlp1-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T12_M4_seed42_120k-layers2-heads-1-1-mlp1-1-d512-lr1e-3-wd0.2-ep500
)
for run_name in "${RUNS[@]}"; do
    best_ckpt="${CKPT_DIR}/${run_name}/${run_name}_best.pt"
    [ ! -f "$best_ckpt" ] && echo "SKIP: $run_name" && continue
    # Parse dmlp{N} from run_name; old ckpts didn't save ffn_dim_list in cfg, so we override.
    dmlp_val=$(echo "$run_name" | grep -oP 'dmlp\K[0-9]+' || true)
    nlayers=$(echo "$run_name" | grep -oP 'layers\K[0-9]+')
    extra_args=""
    if [ -n "$dmlp_val" ]; then
        extra_args="--ffn_dim_list $(yes $dmlp_val | head -n $nlayers | tr '\n' ' ')"
    fi
    output=$(python eval.py --checkpoint_path "$best_ckpt" --test_path "$TEST_PATH" --cell_width $CELL_WIDTH --device cuda $extra_args 2>/dev/null)
    zer=$(echo "$output" | grep "Zero-error" | awk '{print $2}')
    acc=$(echo "$output" | grep "Cell acc" | awk '{print $3}')
    [ -n "$zer" ] && echo "$zer $acc $run_name" >> $RESULTS_FILE
done

# === L16 T10 M8 ===
TEST_PATH=/share/dean/mx253/icl_ca/eca_data/L16_T10_M8_seed42_120k/mixed_test.npz
CELL_WIDTH=15
RUNS=(
    ca-cmixed-L16_T10_M8_seed42_120k-layers2-heads-3-1-mlp1-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M8_seed42_120k-layers2-heads-1-1-mlp1-1-d512-lr1e-3-wd0.2-ep500
)
for run_name in "${RUNS[@]}"; do
    best_ckpt="${CKPT_DIR}/${run_name}/${run_name}_best.pt"
    [ ! -f "$best_ckpt" ] && echo "SKIP: $run_name" && continue
    # Parse dmlp{N} from run_name; old ckpts didn't save ffn_dim_list in cfg, so we override.
    dmlp_val=$(echo "$run_name" | grep -oP 'dmlp\K[0-9]+' || true)
    nlayers=$(echo "$run_name" | grep -oP 'layers\K[0-9]+')
    extra_args=""
    if [ -n "$dmlp_val" ]; then
        extra_args="--ffn_dim_list $(yes $dmlp_val | head -n $nlayers | tr '\n' ' ')"
    fi
    output=$(python eval.py --checkpoint_path "$best_ckpt" --test_path "$TEST_PATH" --cell_width $CELL_WIDTH --device cuda $extra_args 2>/dev/null)
    zer=$(echo "$output" | grep "Zero-error" | awk '{print $2}')
    acc=$(echo "$output" | grep "Cell acc" | awk '{print $3}')
    [ -n "$zer" ] && echo "$zer $acc $run_name" >> $RESULTS_FILE
done

# === L16 T10 M6 ===
TEST_PATH=/share/dean/mx253/icl_ca/eca_data/L16_T10_M6_seed42_120k/mixed_test.npz
CELL_WIDTH=15
RUNS=(
    ca-cmixed-L16_T10_M6_seed42_120k-layers2-heads-3-1-mlp1-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L16_T10_M6_seed42_120k-layers2-heads-1-1-mlp1-1-d512-lr1e-3-wd0.2-ep500
)
for run_name in "${RUNS[@]}"; do
    best_ckpt="${CKPT_DIR}/${run_name}/${run_name}_best.pt"
    [ ! -f "$best_ckpt" ] && echo "SKIP: $run_name" && continue
    # Parse dmlp{N} from run_name; old ckpts didn't save ffn_dim_list in cfg, so we override.
    dmlp_val=$(echo "$run_name" | grep -oP 'dmlp\K[0-9]+' || true)
    nlayers=$(echo "$run_name" | grep -oP 'layers\K[0-9]+')
    extra_args=""
    if [ -n "$dmlp_val" ]; then
        extra_args="--ffn_dim_list $(yes $dmlp_val | head -n $nlayers | tr '\n' ' ')"
    fi
    output=$(python eval.py --checkpoint_path "$best_ckpt" --test_path "$TEST_PATH" --cell_width $CELL_WIDTH --device cuda $extra_args 2>/dev/null)
    zer=$(echo "$output" | grep "Zero-error" | awk '{print $2}')
    acc=$(echo "$output" | grep "Cell acc" | awk '{print $3}')
    [ -n "$zer" ] && echo "$zer $acc $run_name" >> $RESULTS_FILE
done

# === L32 ===
TEST_PATH=/share/dean/mx253/icl_ca/eca_data/L32_T5_M2_seed42_120k/mixed_test.npz
CELL_WIDTH=31
RUNS_L32=(
    ca-cmixed-L32_T5_M2_seed42_120k-layers2-heads-3-1-mlp1-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L32_T5_M2_seed42_120k-layers2-heads-1-1-mlp1-1-d512-lr1e-3-wd0.2-ep500
)
for run_name in "${RUNS_L32[@]}"; do
    best_ckpt="${CKPT_DIR}/${run_name}/${run_name}_best.pt"
    [ ! -f "$best_ckpt" ] && echo "SKIP: $run_name" && continue
    # Parse dmlp{N} from run_name; old ckpts didn't save ffn_dim_list in cfg, so we override.
    dmlp_val=$(echo "$run_name" | grep -oP 'dmlp\K[0-9]+' || true)
    nlayers=$(echo "$run_name" | grep -oP 'layers\K[0-9]+')
    extra_args=""
    if [ -n "$dmlp_val" ]; then
        extra_args="--ffn_dim_list $(yes $dmlp_val | head -n $nlayers | tr '\n' ' ')"
    fi
    output=$(python eval.py --checkpoint_path "$best_ckpt" --test_path "$TEST_PATH" --cell_width $CELL_WIDTH --device cuda $extra_args 2>/dev/null)
    zer=$(echo "$output" | grep "Zero-error" | awk '{print $2}')
    acc=$(echo "$output" | grep "Cell acc" | awk '{print $3}')
    [ -n "$zer" ] && echo "$zer $acc $run_name" >> $RESULTS_FILE
done

# === L10 T20 M8 ===
TEST_PATH=/share/dean/mx253/icl_ca/eca_data/L10_T20_M8_seed42_120k/mixed_test.npz
CELL_WIDTH=9
RUNS=(
    ca-cmixed-L10_T20_M8_seed42_120k-layers2-heads-1-1-mlp1-1-d512-lr1e-3-wd0.2-ep500
    ca-cmixed-L10_T20_M8_seed42_120k-layers2-heads-3-1-mlp1-1-d384-lr1e-3-wd0.2-ep500
    ca-cmixed-L10_T20_M8_seed42_120k-layers2-heads-1-1-mlp1-0-d512-lr1e-3-wd0.2-ep500
)
for run_name in "${RUNS[@]}"; do
    best_ckpt="${CKPT_DIR}/${run_name}/${run_name}_best.pt"
    [ ! -f "$best_ckpt" ] && echo "SKIP: $run_name" && continue
    # Parse dmlp{N} from run_name; old ckpts didn't save ffn_dim_list in cfg, so we override.
    dmlp_val=$(echo "$run_name" | grep -oP 'dmlp\K[0-9]+' || true)
    nlayers=$(echo "$run_name" | grep -oP 'layers\K[0-9]+')
    extra_args=""
    if [ -n "$dmlp_val" ]; then
        extra_args="--ffn_dim_list $(yes $dmlp_val | head -n $nlayers | tr '\n' ' ')"
    fi
    output=$(python eval.py --checkpoint_path "$best_ckpt" --test_path "$TEST_PATH" --cell_width $CELL_WIDTH --device cuda $extra_args 2>/dev/null)
    zer=$(echo "$output" | grep "Zero-error" | awk '{print $2}')
    acc=$(echo "$output" | grep "Cell acc" | awk '{print $3}')
    [ -n "$zer" ] && echo "$zer $acc $run_name" >> $RESULTS_FILE
done

# === L10 T16 M6 ===
TEST_PATH=/share/dean/mx253/icl_ca/eca_data/L10_T16_M6_seed42_120k/mixed_test.npz
CELL_WIDTH=9
RUNS=(
    ca-cmixed-L10_T16_M6_seed42_120k-layers2-heads-1-1-mlp1-1-d512-lr1e-3-wd0.2-ep500
    ca-cmixed-L10_T16_M6_seed42_120k-layers2-heads-3-1-mlp1-1-d384-lr1e-3-wd0.2-ep500
)
for run_name in "${RUNS[@]}"; do
    best_ckpt="${CKPT_DIR}/${run_name}/${run_name}_best.pt"
    [ ! -f "$best_ckpt" ] && echo "SKIP: $run_name" && continue
    # Parse dmlp{N} from run_name; old ckpts didn't save ffn_dim_list in cfg, so we override.
    dmlp_val=$(echo "$run_name" | grep -oP 'dmlp\K[0-9]+' || true)
    nlayers=$(echo "$run_name" | grep -oP 'layers\K[0-9]+')
    extra_args=""
    if [ -n "$dmlp_val" ]; then
        extra_args="--ffn_dim_list $(yes $dmlp_val | head -n $nlayers | tr '\n' ' ')"
    fi
    output=$(python eval.py --checkpoint_path "$best_ckpt" --test_path "$TEST_PATH" --cell_width $CELL_WIDTH --device cuda $extra_args 2>/dev/null)
    zer=$(echo "$output" | grep "Zero-error" | awk '{print $2}')
    acc=$(echo "$output" | grep "Cell acc" | awk '{print $3}')
    [ -n "$zer" ] && echo "$zer $acc $run_name" >> $RESULTS_FILE
done

echo ""
echo "===== Results sorted by zero-error rate ====="
printf "%-12s %-12s %s\n" "Zero-Error" "Cell-Acc" "Run"
echo "-----------------------------------------------"
sort -rn $RESULTS_FILE | while read zer acc name; do
    printf "%-12s %-12s %s\n" "$zer" "$acc" "$name"
done