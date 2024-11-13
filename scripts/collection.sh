# run from inside data_creation folder, so everything is pretty local
export PYTHONWARNINGS=ignore

cd data_creation # so we get in the same directory as the data

NUM_GOALS=$1
ITERATIONS=$2
SEED=$3
ENV=$4

source activate habitat
module load ffmpeg
# replace this with your python path
/scratch/bjb3az/.conda/envs/habitat/bin/python data_collection.py \
    --gesture \
    --num_goals ${NUM_GOALS} \
    --iterations ${ITERATIONS} \
    --seed ${SEED} \
    --env ${ENV}