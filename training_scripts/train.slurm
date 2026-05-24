#!/bin/bash
#SBATCH --job-name=ca-L16-mix
#SBATCH --partition=aimi,dean,default_partition
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/home/mx253/slurm_output/icl_ca/ca_L16_mix_%j.out
#SBATCH --error=/home/mx253/slurm_output/icl_ca/ca_L16_mix_%j.err
#SBATCH --exclude=dean-compute-02

cd ~/CA_Transformer
export WANDB_ENTITY=menghan-xu-cornell-university

# ========== Config ==========
CLASS=mixed  # mixed or 1,2,3,4
DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T10_M4_seed42_30k

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
HIDDEN_SIZE=128
HEADS_LIST="2 4 4 2"
EMB_DROPOUT=0.0
DROPOUT=0.0
BATCH_SIZE=1024
EPOCHS=248
LR=1e-3
WEIGHT_DECAY=0.2
WARMUP_RATIO=0.004
SCHEDULER=cosine
PROJECT_NAME=eca_transformer
SAVE_EVERY=16
EVAL_EVERY=8
CELL_WIDTH=63
ANALYZE_EVERY=64
HEADS_ARRAY=($HEADS_LIST)
NUM_LAYERS=${#HEADS_ARRAY[@]}
# =============================

HEADS_STR=${HEADS_LIST// /-}
DATA_NAME=$(basename $DATA_DIR)
RUN_NAME="ca-c${CLASS}-${DATA_NAME}-layers${NUM_LAYERS}-heads-${HEADS_STR}-d${HIDDEN_SIZE}-lr${LR}-wd${WEIGHT_DECAY}"
mkdir -p $SAVE_DIR/$RUN_NAME

python train.py \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --save_dir $SAVE_DIR \
    --model_type $MODEL_TYPE \
    --seed $SEED \
    --hidden_size $HIDDEN_SIZE \
    --heads_list $HEADS_LIST \
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