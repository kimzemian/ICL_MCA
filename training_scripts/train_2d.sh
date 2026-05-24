#!/bin/bash
export WANDB_ENTITY=menghan-xu-cornell-university
# ========== Config ==========
DATA_DIR=/share/dean/mx253/icl_ca/eca_data/VN_L6x6_T12_M8_rules200+50_rowsep_seed42_120k
TRAIN_PATH=${DATA_DIR}/train.npz
TEST_PATH=${DATA_DIR}/test.npz
SAVE_DIR=/share/dean/mx253/icl_ca/checkpoint
MODEL_TYPE=transformer
SEED=42
HIDDEN_SIZE=1200
HEADS_LIST="5 1"
MLP_LIST="1 1"
VOCAB_SIZE=4
EMB_DROPOUT=0.0
DROPOUT=0.0
BATCH_SIZE=512
EPOCHS=500
LR=1e-3
WEIGHT_DECAY=0.2
WARMUP_RATIO=0.004
SCHEDULER=cosine
PROJECT_NAME=eca_transformer
GPU=0
SAVE_EVERY=16
EVAL_EVERY=8
ANALYZE_EVERY=9999
# =============================

HEADS_ARRAY=($HEADS_LIST)
NUM_LAYERS=${#HEADS_ARRAY[@]}
HEADS_STR=${HEADS_LIST// /-}
MLP_STR=${MLP_LIST// /-}
DATA_NAME=$(basename $DATA_DIR)
RUN_NAME="ca2d-${DATA_NAME}-layers${NUM_LAYERS}-heads-${HEADS_STR}-mlp${MLP_STR}-d${HIDDEN_SIZE}-lr1e-3-wd${WEIGHT_DECAY}-ep${EPOCHS}"
mkdir -p $SAVE_DIR/$RUN_NAME

echo "=== 2D CA Training ==="
echo "Data: $DATA_DIR"
echo "Vocab: $VOCAB_SIZE"
echo "Heads: $HEADS_LIST, MLP: $MLP_LIST, d=$HIDDEN_SIZE"
echo "Run: $RUN_NAME"
echo "GPU: $GPU"
echo ""

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --save_dir $SAVE_DIR \
    --model_type $MODEL_TYPE \
    --seed $SEED \
    --hidden_size $HIDDEN_SIZE \
    --heads_list $HEADS_LIST \
    --use_mlp_list $MLP_LIST \
    --vocab_size $VOCAB_SIZE \
    --emb_dropout $EMB_DROPOUT \
    --dropout $DROPOUT \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --scheduler $SCHEDULER \
    --project_name $PROJECT_NAME \
    --run_name $RUN_NAME \
    --save_every $SAVE_EVERY \
    --eval_every $EVAL_EVERY \
    --analyze_every $ANALYZE_EVERY