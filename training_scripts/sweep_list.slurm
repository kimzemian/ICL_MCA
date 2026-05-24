#!/bin/bash
#SBATCH --job-name=ca-2layer
#SBATCH --partition=dean,default_partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/mx253/slurm_output/icl_ca/2layer_%A_%a.out
#SBATCH --error=/home/mx253/slurm_output/icl_ca/2layer_%A_%a.err
#SBATCH --array=0-10
#SBATCH --requeue

eval "$(conda shell.bash hook)"
conda activate eca
cd ~/CA_Transformer
export WANDB_ENTITY=menghan-xu-cornell-university

DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T10_M4_seed42_30k
TRAIN_PATH=${DATA_DIR}/mixed_train.npz
TEST_PATH=${DATA_DIR}/mixed_test.npz
SAVE_DIR=/share/dean/mx253/icl_ca/checkpoint
HIDDEN_SIZE=256
EPOCHS=500

L_VAL=$(basename $DATA_DIR | grep -oP 'L\K[0-9]+')
CELL_WIDTH=$((L_VAL - 1))

case $SLURM_ARRAY_TASK_ID in
    0) HEADS_LIST="1 1"; MLP_LIST="1 1"; WD=0.2; HIDDEN_SIZE=256 ;;
    1) HEADS_LIST="1 1"; MLP_LIST="1 1"; WD=0.2; HIDDEN_SIZE=128 ;;
    2) HEADS_LIST="1 1"; MLP_LIST="1 1"; WD=0.2; HIDDEN_SIZE=64 ;;
    3) HEADS_LIST="1 1"; MLP_LIST="1 1"; WD=0.2; HIDDEN_SIZE=32 ;;
    4) HEADS_LIST="1 1"; MLP_LIST="1 1"; WD=0.2; HIDDEN_SIZE=16 ;;
    5) HEADS_LIST="3 1"; HIDDEN_SIZE=192; WD=0.2; MLP_LIST="1 1" ;;
    6) HEADS_LIST="3 1"; HIDDEN_SIZE=96; WD=0.2; MLP_LIST="1 1" ;;
    7) HEADS_LIST="3 1"; HIDDEN_SIZE=48; WD=0.2; MLP_LIST="1 1" ;;
    8) HEADS_LIST="3 1"; HIDDEN_SIZE=24; WD=0.2; MLP_LIST="1 1" ;;
    9) HEADS_LIST="3 1"; HIDDEN_SIZE=12; WD=0.2; MLP_LIST="1 1" ;;
    10) HEADS_LIST="1 1"; MLP_LIST="1 1"; WD=0.2; HIDDEN_SIZE=8 ;;
esac

HEADS_ARRAY=($HEADS_LIST)
NUM_LAYERS=${#HEADS_ARRAY[@]}
HEADS_STR=${HEADS_LIST// /-}
DATA_NAME=$(basename $DATA_DIR)
MLP_STR=${MLP_LIST// /-}
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