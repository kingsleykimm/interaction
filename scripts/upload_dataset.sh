
VIDEO_FOLDER=$1
NUM_FRAMES=$2

# note -- you are going to call this from the root directory so keep that in mind when determing where the video_folder will be
/scratch/bjb3az/.conda/envs/llava/bin/python upload_dataset.py --dataset_path ${VIDEO_FOLDER} --num_frames ${NUM_FRAMES}