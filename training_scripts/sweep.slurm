#!/bin/bash
#SBATCH --job-name=ca-width-scan
#SBATCH --partition=dean,default_partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/mx253/slurm_output/icl_ca/width_scan_%A_%a.out
#SBATCH --error=/home/mx253/slurm_output/icl_ca/width_scan_%A_%a.err
#SBATCH --array=0-9
#SBATCH --requeue

eval "$(conda shell.bash hook)"
conda activate eca
cd ~/CA_Transformer
export WANDB_ENTITY=menghan-xu-cornell-university

SAVE_DIR=/share/dean/mx253/icl_ca/checkpoint
MLP_LIST="1 1"
WD=0.2
EPOCHS=500

# 6 jobs: 3 widths x 2 model configs (1-1 d512, 3-1 d384)
case $SLURM_ARRAY_TASK_ID in
    0) DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T10_M6_seed42_120k;  HEADS_LIST="1 1"; HIDDEN_SIZE=512 ;;
    1) DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T10_M6_seed42_120k;  HEADS_LIST="3 1"; HIDDEN_SIZE=384 ;;
    2) DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T10_M8_seed42_120k;  HEADS_LIST="1 1"; HIDDEN_SIZE=512 ;;
    3) DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T10_M8_seed42_120k;  HEADS_LIST="3 1"; HIDDEN_SIZE=384 ;;
    4) DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T12_M4_seed42_120k;  HEADS_LIST="1 1"; HIDDEN_SIZE=512 ;;
    5) DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T12_M4_seed42_120k;  HEADS_LIST="3 1"; HIDDEN_SIZE=384 ;;
    6) DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T14_M4_seed42_120k;  HEADS_LIST="1 1"; HIDDEN_SIZE=512 ;;
    7) DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T14_M4_seed42_120k;  HEADS_LIST="3 1"; HIDDEN_SIZE=384 ;;
    8) DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T16_M4_seed42_120k;  HEADS_LIST="1 1"; HIDDEN_SIZE=512 ;;
    9) DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T16_M4_seed42_120k;  HEADS_LIST="3 1"; HIDDEN_SIZE=384 ;;
esac

TRAIN_PATH=${DATA_DIR}/mixed_train.npz
TEST_PATH=${DATA_DIR}/mixed_test.npz

L_VAL=$(basename $DATA_DIR | grep -oP 'L\K[0-9]+')
CELL_WIDTH=$((L_VAL - 1))

HEADS_ARRAY=($HEADS_LIST)
NUM_LAYERS=${#HEADS_ARRAY[@]}
HEADS_STR=${HEADS_LIST// /-}
MLP_STR=${MLP_LIST// /-}
DATA_NAME=$(basename $DATA_DIR)
RUN_NAME="ca-cmixed-${DATA_NAME}-layers${NUM_LAYERS}-heads-${HEADS_STR}-mlp${MLP_STR}-d${HIDDEN_SIZE}-lr1e-3-wd${WD}-ep${EPOCHS}"
mkdir -p $SAVE_DIR/$RUN_NAME

python train.py \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --save_dir $SAVE_DIR \
    --hidden_size $HIDDEN_SIZE \
    --heads_list $HEADS_LIST \
    --use_mlp_list $MLP_LIST \
    --batch_size 1024 \
    --epochs $EPOCHS \
    --lr 1e-3 \
    --weight_decay $WD \
    --warmup_ratio 0.004 \
    --scheduler cosine \
    --project_name eca_transformer \
    --run_name $RUN_NAME \
    --save_every 32 \
    --eval_every 8 \
    --cell_width $CELL_WIDTH \
    --analyze_every 64