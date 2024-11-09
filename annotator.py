import os
import csv
from transformers import AutoProcessor, AutoModelForCausalLM
from decord import VideoReader, cpu
import numpy as np
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--num_frames", type=int, help = "number of frames to split video into")
parser.add_argument("--video_dir", type=str, help="which video directory we're looking into")
# cite: https://github.com/LLaVA-VL/LLaVA-NeXT/blob/inference/playground/demo/video_demo.py

INFERENCE_PROMPT = """ 
Imagine you are a robot at home, operating inside an apartment or household along with another agent, a human. You are a confident robot who knows
      how to operate in these environments.
      You are going to receive a video showing an interaction between you and the human, where you run into each other and need to make a decision.
      The human might perform some sort of action that signals you to go ahead, but regardless, you must give the next action you will take, this is mandatory. Focus only on the gestures/movement
      the human is making, not on anything else in the environment. 
      Please explain step-by-step why you chose to do the certain actions and be concise in your response but include any essential information.
"""

def load_video(video_path, num_frames):
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
    return spare_frames, frame_time, video_time

def run_inference_labelling(args):
    all_video_paths = []
    video_dir = args.video_dir
    processor = AutoProcessor.from_pretrained("lmms-lab/LLaVA-NeXT-Video-32B-Qwen")
    model = AutoModelForCausalLM.from_pretrained("lmms-lab/LLaVA-NeXT-Video-32B-Qwen")

    conversation = [
        {
            "role": "system",
            "content" : [
                {"type" : "text", "text" : "You are a helpful assistant"}
            ]
        },
        {
            "role" : "user",
            "content" : [
                {"type" : "text", "text" : INFERENCE_PROMPT },
                {"type" : "video"},
            ]
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    

    if os.path.isdir(video_dir):
        # If it's a directory, loop over all files in the directory
        for filename in os.listdir(video_dir):
                    # Load the video file
            cur_video_path = os.path.join(video_dir, f"{filename}")
            all_video_paths.append(cur_video_path)
    data = [
        ["file_name", "model_output"]
    ]
    for video_path in all_video_paths:
        video, frame_time, video_time = load_video(video_path, args.num_frames)
        inputs = processor(text = prompt, videos=video, return_tensors="pt")
        out = model.generate(**inputs)
        outputs = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip() # always returns a list so must 0-index
        data_row = [os.path.split(video_path)[-1], outputs]
        data.append(data_row)
    
    with open(os.path.join(video_dir, "metadata.csv"), "w") as file:
        writer = csv.writer(file)
        writer.writerows(data)


if __name__ == "__main__":
    args = parser.parse_args()
    run_inference_labelling(args)