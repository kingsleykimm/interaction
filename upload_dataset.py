from argparse import ArgumentParser
from datetime import datetime
import numpy as np
from datasets import load_dataset, Video
from decord import VideoReader, cpu
from transformers import LlavaNextVideoProcessor
parser = ArgumentParser()
parser.add_argument('--dataset_path', type=str, help="path/link to the hf or local datset")
parser.add_argument('--num_frames', type=int, help="Number of frames to split videos into")

MASTER_PROMPT = """ 
Imagine you are a robot at home, operating inside an apartment along with another agent, a human. You a robot who understands human intent well and has a firm understanding of social cues and the importance of cooperation betwen humans and robots in shared spaces.
      You encounter a human in the apartment, and you run into each other and you need to make a decision of how you will proceed, this is mandatory. 
      Don't provide any 'ifs' in your answer, be explicit, your only goal is to determine what your next action is, don't worry about any other goals.
      Please explain step-by-step why you chose to do the certain actions and be concise in your response and inclue only essential information.
      Your answer should be formatted like this: ’Action: [action]], Reasoning: [reasoning]’, where action details the next action you will take, and reasoning is your explanation behind that choice.
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
        label = example["model_output"].strip().strip('\n')
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
            return_tensors="pt",
            truncation=True
        )
        
        return batch

    dataset = dataset.map(collate_fn, batched=False, num_proc=1,writer_batch_size=400)
    return dataset

if __name__ == '__main__':
    args = parser.parse_args()
    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", use_fast=False) # this is already the tokenizer
    processor.tokenizer.padding_side = 'right'
    dataset = load_in_dataset(args.dataset_path, processor, args.num_frames)
    dataset = dataset.train_test_split(test_size=0.2)
    format_data = "%d_%m_%y_%H_%M_%S"
    cur_time = datetime.now()
    dataset_name = f"kingsleykim/habitat_videos_{cur_time.strftime(format_data)}"
    # make sure to huggingface-cli login before doing this
    dataset.push_to_hub(dataset_name)

    # we upload both so we can download both as well

