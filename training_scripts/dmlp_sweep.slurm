#!/bin/bash
#SBATCH --job-name=ca-dmlp
#SBATCH --partition=dean,default_partition
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03,genai-large-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/mx253/slurm_output/icl_ca/dmlp_%A_%a.out
#SBATCH --error=/home/mx253/slurm_output/icl_ca/dmlp_%A_%a.err
#SBATCH --array=0-8
#SBATCH --requeue

eval "$(conda shell.bash hook)"
conda activate eca
cd ~/CA_Transformer
export WANDB_ENTITY=menghan-xu-cornell-university

DATA_DIR=/share/dean/mx253/icl_ca/eca_data/L16_T10_M4_seed42_30k
TRAIN_PATH=${DATA_DIR}/mixed_train.npz
TEST_PATH=${DATA_DIR}/mixed_test.npz
SAVE_DIR=/share/dean/mx253/icl_ca/checkpoint
EPOCHS=300
HEADS_LIST="1 1"
MLP_LIST="1 1"
WD=0.2

L_VAL=$(basename $DATA_DIR | grep -oP 'L\K[0-9]+')
CELL_WIDTH=$((L_VAL - 1))

# (d_model, d_mlp) combinations
case $SLURM_ARRAY_TASK_ID in
    0) HIDDEN_SIZE=16;  D_MLP=32  ;;
    1) HIDDEN_SIZE=15;  D_MLP=32  ;;
    2) HIDDEN_SIZE=32;  D_MLP=128 ;;
    3) HIDDEN_SIZE=64;  D_MLP=128 ;;
    4) HIDDEN_SIZE=128; D_MLP=128 ;;
    5) HIDDEN_SIZE=128; D_MLP=64  ;;
    6) HIDDEN_SIZE=128; D_MLP=32  ;;
    7) HIDDEN_SIZE=128; D_MLP=16  ;;
    8) HIDDEN_SIZE=128; D_MLP=8   ;;
esac

FFN_DIM_LIST="$D_MLP $D_MLP"

HEADS_ARRAY=($HEADS_LIST)
NUM_LAYERS=${#HEADS_ARRAY[@]}
HEADS_STR=${HEADS_LIST// /-}
MLP_STR=${MLP_LIST// /-}
DATA_NAME=$(basename $DATA_DIR)
RUN_NAME="ca-cmixed-${DATA_NAME}-layers${NUM_LAYERS}-heads-${HEADS_STR}-mlp${MLP_STR}-d${HIDDEN_SIZE}-dmlp${D_MLP}-lr1e-3-wd${WD}-ep${EPOCHS}"
mkdir -p $SAVE_DIR/$RUN_NAME

python train.py \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --save_dir $SAVE_DIR \
    --hidden_size $HIDDEN_SIZE \
    --heads_list $HEADS_LIST \
    --use_mlp_list $MLP_LIST \
    --ffn_dim_list $FFN_DIM_LIST \
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
