from argparse import ArgumentParser
from datetime import datetime
import numpy as np
from datasets import load_dataset, Video
from transformers import AutoProcessor
from decord import VideoReader, cpu
from qwen_vl_utils import process_vision_info
import os
from prompts import BETTER_FORMATTED_ICL_PROMPT, COT_FIRST_EXAMPLE_ANSWER, COT_SECOND_EXAMPLE_ANSWER, example_gesture_video, example_no_gesture_video
parser = ArgumentParser()
parser.add_argument('--dataset_path', type=str, help="path/link to the hf or local datset")
parser.add_argument('--num_frames', type=int, default=12, help="Number of frames to split videos into")

BETTER_FORMATTED_ICL_PROMPT = """
Context: You are a robot operating inside an apartment with a human. You are a confident and assertive robot who understands human intent well and takes a human's intent into account when determing your course of action. Human intent can be defined as
the human's arm movement, body language and walking movements/direction. You are engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.


Instructions:
- Identify if the human is making a gesture in your direction or not. This must be the first sentence of your answer
- Assume you and the human have the same target destination
- Determine your next action to avoid your trajectory and the human's trajectory colliding, deducing the human's intent through gestures and body language. 
- Do not communicate with the human. Focus only on what the human is communicating to you. 
- Ignore any objects in the environment.
- Keep in mind you and the human have the same destination
- Output your answer in the format: (### [Choose between Walk Left, Walk Right, Walk Straight, Stay.])
"""

COT_FIRST_EXAMPLE_ANSWER = """
The human makes a visible gesture in my general direction, and it seems like they are motioning for me to go first, towards the right direction.
Both of us are trying to go to the right, but the human gestured for me to go first. I believe the human intends for me to go first and they will follow after.
At the moment that I encounter the human, it appears our trajectories will soon overlap, causing collision. To resolve the collision, the human gestures to me to go first towards my right. We are both intending to go to the right, but the human gestured for me to go first.
To avoid collision, I should start going to my right, since that is where the human gestured. I will assume that the human will follow after me, since we have the same destination.
### Walk Right
"""

COT_SECOND_EXAMPLE_ANSWER = """
The human doesn't seem to be making a visible gesture in my direction.
I don't think the human is making a gesture, but based off their body language, they are walking towards the doorway in front of them, which is my destination as well.
At the moment that I encounter the human, it appears they are trying to enter the room on the left, which is also my destination. The human does not seem to be stopping their path and will continue walking.
Based off the human's intent, they are not going to slow down, and they will keep walking into the room. If I continue to walk as well, there is a high chance of collision. THus, I will wait for the human to enter the room before continuing my path and walking into the room as well, this way collision is avoided.
### Wait
"""


# Remember to take these out of the training set next time
example_gesture_video = "/scratch/bjb3az/interaction/n_shot_examples/target_002_master_chef_can_:0000_iteration_3_gesture_True_2024-11-17_19:15:20_seed_66.mp4"
example_no_gesture_video = "/scratch/bjb3az/interaction/n_shot_examples/target_009_gelatin_box_:0000_iteration_27_gesture_False_2024-11-17_13:57:55_seed_65.mp4"

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
        video_path = example['video']['path']
        # example["video"] = video_clip

        conversation = [{
            "role" : "user",
            "content" : [
            {
                "type" : "text", "text" : BETTER_FORMATTED_ICL_PROMPT,
            },
            {
                "type" : "video", "video" : example_gesture_video, 'nframes' : num_frames
            },
            {
                "type" : "text", "text" : COT_FIRST_EXAMPLE_ANSWER,
            },
            {
                "type" : "video", "video" : example_no_gesture_video,
            },
            {
                "type": "text", "text" : COT_SECOND_EXAMPLE_ANSWER, 'nframes' : num_frames
            },
            {
                "type" : "text", "text" : "New scenario:"
            },
            {
                "type" : "video", "video" : video_path, 'nframes' : num_frames
            }
            ]},
            {
                "role" : "assistant",
                "content" : [
                    {
                        "type" : "text", "text" : "Let's think step by step: "
                    }
                ]
            }
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, continue_final_message=True)
        image_inputs, vids_inputs = process_vision_info(conversation)
        # we can use paligemma + intern which are vision language models only for image but it shouldn't be that differnt
        # Look into whether we need different methods for different processing methods
        # process vision info just retuns the actual video input split up by num_frames, as a list of torch.Tensors, so it's pretty easy to insert into any model as long as we know the input names
        batch = processor(
            text=[text],
            videos=vids_inputs, # this turns it into pixel_inputs so we can just preserve that in the dataset
            return_tensors="pt",
            padding=True,
        )
        # need to convert model outputs as well
        outputs = processor(text=[label], return_tensors="pt", padding=True)
        print(outputs)
        return {'model_inputs' : batch, 'goal_output' : outputs}
    def remove_lines(example):
        stripped = example['model_output'].replace('\n', " ")
        model_output = stripped
        vid_name = example['video']['path']
        video = VideoReader(os.path.join(file_path, vid_name))
        print(video)
        return {
            "file_name" : vid_name,
            "video" : video,
            "model_output" : model_output
        }
    # dataset = dataset.map(collate_fn, batched=False, num_proc=1,writer_batch_size=400)
    dataset = dataset.map(remove_lines, batched=False)
    return dataset

if __name__ == '__main__':
    args = parser.parse_args()
    # processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", use_fast=False) # this is already the tokenizer
    processor = AutoProcessor.from_pretrained("./models/Qwen2-VL-72B-Instruct")
    processor.tokenizer.padding_side = 'right'
    dataset = load_in_dataset(args.dataset_path, processor, args.num_frames)
    dataset = dataset.train_test_split(test_size=0.2)
    print(dataset['train'][0])
    format_data = "%d_%m_%y_%H_%M_%S"
    cur_time = datetime.now()
    dataset_name = f"kingsleykim/habitat_videos"
    # make sure to huggingface-cli login before doing this
    dataset.push_to_hub(dataset_name)

    # we upload both so we can download both as well

