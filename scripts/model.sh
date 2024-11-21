
DATASET_PATH=$1
NUM_FRAMES=$2
MODEL_NAME=$3
IF_TRAIN=$4

module purge

module load cuda/12.2.2
module load apptainer pytorch


if [ "$IF_TRAIN" = True ]; then

/scratch/bjb3az/.conda/envs/habitat/bin/python train.py \
    --dataset_path $DATASET_PATH \
    --num_frames $NUM_FRAMES \

else

/scratch/bjb3az/.conda/envs/habitat/bin/python inference.py \
    --hf_dataset $DATASET_PATH \
    --finetuned $MODEL_NAME \

fi