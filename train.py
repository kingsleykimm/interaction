# Load model directly
import os
import av
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from decord import VideoReader, cpu

USE_LORA = True
MAX_LENGTH = 256
BATCH_SIZE = None
NUM_FRAMES = 32

CHECKPOINT_DIR = '/scratch/bjb3az/interaction/model_ckpt'

parser = ArgumentParser()
parser.add_argument('--dataset_path', type=str, help="path/link to the hf or local datset")
parser.add_argument('--num_frames', type=int, help="Number of frames to split videos into")
parser.add_argument('--ckpt_dir', type=str, help="directory of model we save to")


def load_in_dataset(hf : bool, file_path):
    dataset = None
    if hf: # if loading from huggingface
        dataset = load_dataset(file_path)
    else: # local load
        return load_dataset("videofolder", data_dir=file_path, split="train")
    
def load_video(video_path,num_frames):
    if num_frames == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > num_frames:
        sample_fps = num_frames
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames,frame_time,video_time
    
if __name__ == "__main__":
    load_dataset(false, "good_data")