#!/bin/bash
#SBATCH --job-name=ca-mixedL
#SBATCH --partition=dean,default_partition
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/mx253/slurm_output/icl_ca/mixedL_%A_%a.out
#SBATCH --error=/home/mx253/slurm_output/icl_ca/mixedL_%A_%a.err
#SBATCH --array=0-1
#SBATCH --requeue

eval "$(conda shell.bash hook)"
conda activate eca
cd ~/CA_Transformer
export WANDB_ENTITY=menghan-xu-cornell-university

SAVE_DIR=/share/dean/mx253/icl_ca/checkpoint
DATA_DIR=/share/dean/mx253/icl_ca/eca_data/mixedL_10-16-22-28_T10_M4_seed42_120k
MLP_LIST="1 1"
WD=0.2
EPOCHS=500

TRAIN_PATH=${DATA_DIR}/mixed_train_L10-16-22-28.npz
TEST_PATH=${DATA_DIR}/mixed_test_seen_L10-16-22-28.npz
TEST_UNSEEN_PATH=${DATA_DIR}/mixed_test_unseen_L13-19-25.npz



case $SLURM_ARRAY_TASK_ID in
    0) HEADS_LIST="1 1"; HIDDEN_SIZE=512 ;;
    1) HEADS_LIST="3 1"; HIDDEN_SIZE=384 ;;
esac

HEADS_ARRAY=($HEADS_LIST)
NUM_LAYERS=${#HEADS_ARRAY[@]}
HEADS_STR=${HEADS_LIST// /-}
MLP_STR=${MLP_LIST// /-}
RUN_NAME="ca-mixedL-10-16-22-28-layers${NUM_LAYERS}-heads-${HEADS_STR}-mlp${MLP_STR}-d${HIDDEN_SIZE}-lr1e-3-wd${WD}-ep${EPOCHS}-v2"
mkdir -p $SAVE_DIR/$RUN_NAME

python train.py \
    --mixed_L \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --test_path_unseen $TEST_UNSEEN_PATH \
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
    --analyze_every 64