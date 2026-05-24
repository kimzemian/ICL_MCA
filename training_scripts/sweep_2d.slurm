#!/bin/bash
#SBATCH --job-name=ca2d-500k-v4
#SBATCH --partition=gpu
##SBATCH --nodelist=nikola-compute-[11-18] 
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mx253/slurm_output/icl_ca/ca2d_500k_v4_%A_%a.out
#SBATCH --error=/home/mx253/slurm_output/icl_ca/ca2d_500k_v4_%A_%a.err
#SBATCH --array=0-3
#SBATCH --requeue

eval "$(conda shell.bash hook)"
conda activate eca
cd ~/CA_Transformer
export WANDB_ENTITY=menghan-xu-cornell-university

SAVE_DIR=/share/dean/mx253/icl_ca/checkpoint
WD=0.2
DATA_DIR=/share/dean/mx253/icl_ca/eca_data/VN_L6x6_T12_M8_rules200+50_aug_rowsep_seed42_500k_new
VOCAB_SIZE=4

case $SLURM_ARRAY_TASK_ID in
    # 4-layer, lr sweep
    0) HEADS_LIST="5 1 1 1"; HIDDEN_SIZE=640; LR=1e-3; EPOCHS=500; BS=256 ;;
    1) HEADS_LIST="5 1 1 1"; HIDDEN_SIZE=640; LR=2e-3; EPOCHS=500; BS=256 ;;
    # Wide 2-layer
    2) HEADS_LIST="5 1"; HIDDEN_SIZE=1280; LR=3e-3; EPOCHS=300; BS=256 ;;
    3) HEADS_LIST="5 1"; HIDDEN_SIZE=1280; LR=1e-3; EPOCHS=300; BS=256 ;;
esac

MLP_ARRAY=()
for h in $HEADS_LIST; do MLP_ARRAY+=(1); done
MLP_LIST="${MLP_ARRAY[*]}"

TRAIN_PATH=${DATA_DIR}/train.npz
VAL_PATH=${DATA_DIR}/val.npz
TEST_PATH=${DATA_DIR}/test.npz

HEADS_ARRAY=($HEADS_LIST)
NUM_LAYERS=${#HEADS_ARRAY[@]}
HEADS_STR=${HEADS_LIST// /-}
MLP_STR=${MLP_LIST// /-}
RUN_NAME="ca2d-500k_v4-layers${NUM_LAYERS}-heads-${HEADS_STR}-mlp${MLP_STR}-d${HIDDEN_SIZE}-lr${LR}-wd${WD}-ep${EPOCHS}"
mkdir -p $SAVE_DIR/$RUN_NAME

echo "=== Job $SLURM_ARRAY_TASK_ID ==="
echo "Heads: $HEADS_LIST, MLP: $MLP_LIST, d=$HIDDEN_SIZE"
echo "BS=$BS, LR=$LR, Epochs=$EPOCHS"
echo "Run: $RUN_NAME"

python train.py \
    --train_path $TRAIN_PATH \
    --val_path $VAL_PATH \
    --test_path $TEST_PATH \
    --save_dir $SAVE_DIR \
    --hidden_size $HIDDEN_SIZE \
    --heads_list $HEADS_LIST \
    --use_mlp_list $MLP_LIST \
    --vocab_size $VOCAB_SIZE \
    --batch_size $BS \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WD \
    --warmup_ratio 0.004 \
    --scheduler cosine \
    --project_name eca_transformer \
    --run_name $RUN_NAME \
    --save_every 32 \
    --eval_every 16 \
    --analyze_every 9999
