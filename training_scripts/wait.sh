cd ~/CA_Transformer
export WANDB_ENTITY=menghan-xu-cornell-university

echo "Queue started at $(date). Waiting for current jobs..."

while pgrep -u mx253 -f "python train.py" > /dev/null; do
    echo "$(date): still running ($(pgrep -u mx253 -f "python train.py" | wc -l) processes)"
    sleep 1800
done

echo "$(date): All done. Starting V=3 experiments."

# V=3, 3+1 heads
RUN_NAME=ca1d-V3_L32_T12_M8_rules200+50_seed42_120k-layers2-heads-3-1-mlp1-1-d600-lr1e-3-wd0.2-ep500
mkdir -p /share/dean/mx253/icl_ca/checkpoint/$RUN_NAME
python train.py \
    --train_path /share/dean/mx253/icl_ca/eca_data/V3_L32_T12_M8_rules200+50_seed42_120k/train.npz \
    --test_path /share/dean/mx253/icl_ca/eca_data/V3_L32_T12_M8_rules200+50_seed42_120k/test.npz \
    --save_dir /share/dean/mx253/icl_ca/checkpoint \
    --hidden_size 600 --heads_list 3 1 --use_mlp_list 1 1 --vocab_size 4 \
    --batch_size 256 --epochs 500 --lr 1e-3 --weight_decay 0.2 \
    --warmup_ratio 0.004 --scheduler cosine \
    --project_name eca_transformer --run_name $RUN_NAME \
    --save_every 16 --eval_every 8 --cell_width 31 --analyze_every 9999 \
    > ~/CA_Transformer/log/$RUN_NAME.log 2>&1 &

# V=3, 1+1 heads
RUN_NAME=ca1d-V3_L32_T12_M8_rules200+50_seed42_120k-layers2-heads-1-1-mlp1-1-d600-lr1e-3-wd0.2-ep500
mkdir -p /share/dean/mx253/icl_ca/checkpoint/$RUN_NAME
python train.py \
    --train_path /share/dean/mx253/icl_ca/eca_data/V3_L32_T12_M8_rules200+50_seed42_120k/train.npz \
    --test_path /share/dean/mx253/icl_ca/eca_data/V3_L32_T12_M8_rules200+50_seed42_120k/test.npz \
    --save_dir /share/dean/mx253/icl_ca/checkpoint \
    --hidden_size 600 --heads_list 1 1 --use_mlp_list 1 1 --vocab_size 4 \
    --batch_size 256 --epochs 500 --lr 1e-3 --weight_decay 0.2 \
    --warmup_ratio 0.004 --scheduler cosine \
    --project_name eca_transformer --run_name $RUN_NAME \
    --save_every 16 --eval_every 8 --cell_width 31 --analyze_every 9999 \
    > ~/CA_Transformer/log/$RUN_NAME.log 2>&1 &

wait
echo "$(date): V=3 experiments done."