#!/bin/bash
#SBATCH --job-name=ca-sanity
#SBATCH --partition=dean,default_partition
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03,genai-large-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/mx253/slurm_output/icl_ca/sanity_%A.out
#SBATCH --error=/home/mx253/slurm_output/icl_ca/sanity_%A.err
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
HIDDEN_SIZE=512
HEADS_LIST="1 1"
MLP_LIST="1 1"
WD=0.2

L_VAL=$(basename $DATA_DIR | grep -oP 'L\K[0-9]+')
CELL_WIDTH=$((L_VAL - 1))
DATA_NAME=$(basename $DATA_DIR)
HEADS_STR=${HEADS_LIST// /-}
MLP_STR=${MLP_LIST// /-}
RUN_NAME="ca-cmixed-${DATA_NAME}-layers2-heads-${HEADS_STR}-mlp${MLP_STR}-d${HIDDEN_SIZE}-lr1e-3-wd${WD}-ep${EPOCHS}-sanity"
mkdir -p $SAVE_DIR/$RUN_NAME

# NOTE: not passing --ffn_dim_list — tests backward-compat default (4 * hidden_size = 2048)
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
