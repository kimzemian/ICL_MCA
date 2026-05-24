#!/bin/bash
export WANDB_ENTITY=menghan-xu-cornell-university
# ========== Config ==========
CLASS=mixed
DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L10_T20_M8_seed42_120k

if [ "$CLASS" = "mixed" ]; then
    TRAIN_PATH=${DATA_DIR}/mixed_train.npz
    TEST_PATH=${DATA_DIR}/mixed_test.npz
else
    TRAIN_PATH=${DATA_DIR}/class${CLASS}_train.npz
    TEST_PATH=${DATA_DIR}/class${CLASS}_test.npz
fi
SAVE_DIR=/share/dean/mx253/icl_ca/checkpoint
MODEL_TYPE=transformer
SEED=42
HIDDEN_SIZE=512
HEADS_LIST="1 1"
EMB_DROPOUT=0.0
DROPOUT=0.0
BATCH_SIZE=1024
EPOCHS=500
LR=1e-3
WEIGHT_DECAY=0.2
WARMUP_RATIO=0.004
SCHEDULER=cosine
PROJECT_NAME=eca_transformer
GPU=0
SAVE_EVERY=16
EVAL_EVERY=8
L_VAL=$(basename $DATA_DIR | grep -oP 'L\K[0-9]+')
CELL_WIDTH=$((L_VAL - 1))
ANALYZE_EVERY=64
HEADS_ARRAY=($HEADS_LIST)
NUM_LAYERS=${#HEADS_ARRAY[@]}
MLP_LIST="1 1"
# =============================

HEADS_STR=${HEADS_LIST// /-}
DATA_NAME=$(basename $DATA_DIR)
MLP_STR=${MLP_LIST// /-}
RUN_NAME="ca-cmixed-${DATA_NAME}-layers${NUM_LAYERS}-heads-${HEADS_STR}-mlp${MLP_STR}-d${HIDDEN_SIZE}-lr1e-3-wd${WEIGHT_DECAY}-ep${EPOCHS}"
mkdir -p $SAVE_DIR/$RUN_NAME

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --save_dir $SAVE_DIR \
    --model_type $MODEL_TYPE \
    --seed $SEED \
    --hidden_size $HIDDEN_SIZE \
    --heads_list $HEADS_LIST \
    --use_mlp_list $MLP_LIST \
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
    --cell_width $CELL_WIDTH \
    --analyze_every $ANALYZE_EVERY