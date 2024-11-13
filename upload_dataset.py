from argparse import ArgumentParser
from datetime import datetime
import numpy as np
from datasets import load_dataset, Video
from decord import VideoReader, cpu
from transformers import LlavaNextVideoProcessor
from huggingface_hub import HfApi
parser = ArgumentParser()
parser.add_argument('--dataset_path', type=str, help="path/link to the hf or local datset")
parser.add_argument('--num_frames', type=int, help="Number of frames to split videos into")

MASTER_PROMPT = """ 
Imagine you are a robot at home, operating inside an apartment or household along with another agent, a human. You are a confident robot who knows
      how to operate in these environments.
      You are going to receive a video showing an interaction between you and the human, where you run into each other and need to make a decision.
      The human might perform some sort of action that signals you to go ahead, but regardless, you must give the next action you will take, this is mandatory. Focus only on the gestures/movement
      the human is making, not on anything else in the environment. 
      Please explain step-by-step why you chose to do the certain actions and be concise in your response but include any essential information.
"""

def load_video(video_path,num_frames):
    vr = VideoReader(uri=video_path, ctx=cpu(0)) # you need to install from source to use gpu ctx
    indices = np.arange(0, len(vr), len(vr) / num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def load_in_dataset(file_path, processor, num_frames):
    dataset = load_dataset("videofolder", data_dir=file_path, split="train")
    dataset = dataset.cast_column("video", Video(decode=False)) # to remove the automatic decord
    def collate_fn(example):
    # data format will just have filename and label
        label = example["model_output"]
        video_clip = load_video(example["video"]['path'], num_frames)
        # example["video"] = video_clip

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": MASTER_PROMPT},
                    {"type": "video"},
                    ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": label},
                        ],
            },]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
        batch = processor(
            text=prompt,
            videos=video_clip,
            return_tensors="pt"
        )
        
        return batch

    dataset = dataset.map(collate_fn, batched=False, num_proc=1,writer_batch_size=400)
    return dataset

if __name__ == '__main__':
    args = parser.parse_args()
    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", use_fast=False) # this is already the tokenizer
    dataset = load_in_dataset(args.dataset_path, processor, args.num_frames)
    dataset = dataset.train_test_split(test_size=0.2)
    format_data = "%d_%m_%y_%H_%M_%S"
    cur_time = datetime.now()
    dataset_name = f"kingsleykim/habitat_videos_{cur_time.strftime(format_data)}"
    # make sure to huggingface-cli login before doing this
    dataset.push_to_hub(dataset_name)
    # now need to upload to hf
    api = HfApi()
    api.upload_folder(folder_path=args.dataset_path, repo_id=dataset_name, repo_type='dataset' )

    # we upload both so we can download both as well

