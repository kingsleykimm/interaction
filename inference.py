from argparse import ArgumentParser
import csv
import os
import re
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor, LlavaNextVideoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
import torch
import time
import math
from datasets import load_dataset
from decord import VideoReader, cpu
from collections import defaultdict
from qwen_vl_utils import process_vision_info
import random
from prompts import *
INFERENCE_PROMPT = """ 
Imagine you are a robot at home, operating inside an apartment along with another agent, a human. You a robot who understands human intent well and has a firm understanding of social cues and the importance of cooperation betwen humans and robots in shared spaces.
      You encounter a human in the apartment, and you run into each other and you need to make a decision of how you will proceed, this is mandatory. 
      Don't provide any 'ifs' in your answer, be explicit, your only goal is to determine what your next action is, don't worry about any other goals.
      Please explain step-by-step why you chose to do the certain actions and be concise in your response and inclue only essential information. If you aren't sure, 
      Your answer should be formatted like this: ’Action: [action]], Reasoning: [reasoning]’, where action details the next action you will take, and reasoning is your explanation behind that choice.
"""

parser = ArgumentParser()
parser.add_argument('--hf_dataset', type=str, help='hf dataset to pull from')
parser.add_argument("--finetuned", type=str, help="Finetuned model on HF hub", default="./models/qwen-vl-7b-finetune-1")
parser.add_argument("--base_model", type=str, help="Base model to compare with (local)", default="./models/Qwen2-VL-7B-Instruct")


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        './models/Qwen2-VL-72B-Instruct/' : 80, './models/Qwen2-VL-7B-Instruct/' : 28}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.2))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.8)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['visual'] = 0
    device_map['model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['model.norm'] = 0
    device_map['lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def majority_voting_action(outputs):
    counts = defaultdict(int)
    
    # run regex on here
    for text in outputs:
        print(text)
        print(len(text))
        match = re.search(r"###\s*(.+)", text)
        if match:
            final_action = match.group(1).strip()
            counts[final_action] += 1
        else:
            print("No match found.")
    max_count = 0
    print(counts)
    for action in counts:
        if counts[action] > max_count:
            max_count = counts[action]
    actions_sample = []
    for action in counts:
        if counts[action] == max_count:
            actions_sample.append(action)
    print(actions_sample)
    if len(actions_sample) == 0:
        return "NO ACTION", ""
    fin_action = random.sample(actions_sample, 1)[0]

    final_outputs = []
    for text in outputs:
        match = re.search(r"###\s*(.+)", text)
        if match:
            final_action = match.group(1).strip()
            if final_action == fin_action:
                final_outputs.append(text)
    if len(final_outputs) == 0:
        return "NO ACTION", ""
    return random.sample(final_outputs, 1)[0], fin_action
def load_model(model_path):
    # we are probably going to do full finetuning since 7B * 18 = 126 < 128 GB available
    if "qwen" in model_path.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=split_model(model_path),
            # attn_implementation="flash_attention_2"
            )
    else:
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype= torch.bfloat16,
            # _attn_implementation="flash_attention_2",
            device_map=split_model(model_path),
        )
    return model

def run_inference(dataset, base, finetune, processor):
    # we want to take in a prompt and run inference, but when evaluating we also want to make it take comparisons between our finetuned model and a baseline, whcih will be LLava-7b
    # humans can evaluate the paired responses
    
    data = [
        "video_file", "base_output", "finetuned_output"
    ]
    base_acc, finetuned_acc = 0.0, 0.0
    for ind, item in enumerate(dataset): # do we even need this, or can we do this across the entire dataset?
        start_time = time.time()
        video_path = item['video']['path']
        # video = VideoReader(item['video'])
        # video = torch.from_numpy(video)
        ground_action = None
        match = re.search(r"###\s*(.+)", item['model_output'])
        if match:
            ground_action = match.group(1).strip()
            
        # the format here has the folder it was in as vid_path
        conversation = [
        {
            "role" : "system",
            "content": [
                {"type" : "text", "text" : BETTER_FORMATTED_ICL_PROMPT},
            ]
        },
            
        {
            "role" : "user",
            "content": [
                {"type" : "text", "text" : "Video: "},
                {"type" : "video", "video" : video_path}
            ]
        },
        {
            "role" : "assistant",
            "content" : [{
                
                "type" : "text", "text": "Let's think step by step: "

            }]
        }
        ]
        # conversation = [
        # {
        #     "role" : "system",
        #     "content":  [
        #         {
        #             "type" : "text", "text" : "You are a coding model that understands video. All instructions are in the form of Python comments."
        #         },
        #         {
        #             "type" : "text", "text" : """
                    
        #             \"\"\"Robot task programs.
        #             Robot task programs may use the following functions:
        #             say(message)
        #             ask(question, options)
        #             predict_human_path()
        #             walk(direction, distance)
        #             find_next_location()
        #             does_collision_exist(human_walk_vector)
        #             gesture_performed()

        #             Robot tasks are defined in named functions, with docstrings describing the task.
        #             \"\"\"


        #             # Ask the human a question, and offer a set of specific options for the person to respond. Returns the response selected by the person.
        #             def ask(question : str, options: list[str]) -> str:
        #             ...

        #             # Say the message out loud.
        #             def say(message : str) -> None:
        #             ...

        #             def predict_human_path() -> tuple(int, int):



        #             """
        #         }
        #     ]

        # },
        #     {
        #         "role" : "user",
        #         "content" : {

        #         }
        #     }
        # ]
        # prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info(conversation)
        # base_batch = processor(
        #     text=prompt,
        #     videos=video_inputs,
        #     return_tensors="pt"
        # ).to(base.device)
        # finetuned_batch = processor(
        #     text=prompt,
        #     videos=video_inputs,
        #     return_tensors="pt"
        # ).to(finetune.device)
        # conversation = [{
        #     "role" : "user",
        #     "content" : [
        #     {
        #         "type" : "text", "text" : BETTER_FORMATTED_ICL_PROMPT,
        #     },
        #     {
        #         "type" : "text", "text" : "Video:"
        #     },
        #     {
        #         "type" : "video", "video" : example_gesture_video, 'nframes' : 12
        #     },
        #     {
        #         "type" : "text", "text" : COT_FIRST_EXAMPLE_ANSWER,
        #     },
        #     {
        #         "type" : "text", "text" : "Video:"
        #     },
        #     {
        #         "type" : "video", "video" : example_no_gesture_video,
        #     },
        #     {
        #         "type": "text", "text" : COT_SECOND_EXAMPLE_ANSWER, 'nframes' : 12
        #     },
        #     {
        #         "type" : "text", "text" : "Video:"
        #     },
        #     {
        #         "type" : "video", "video" : vid_name, 'nframes' : 12
        #     }
        #     ]},
        #     {
        #         "role" : "assistant",
        #         "content" : [
        #             {
        #                 "type" : "text", "text" : "Let's think step by step: "
        #             }
        #         ]
        #     }
        # ]
        text = processor.apply_chat_template(conversation, tokenize=False, continue_final_message=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        batch = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt'
        ).to(base.device)
        finetuned_batch = batch.to(finetune.device)
        # do it for both models, add
        # item = item['model_inputs']
        # base_video_clip, finetuned_video_clip = item['pixel_values_videos'].to(base.device), item['pixel_values_videos'].to(finetune.device)
        # base_grid_thw, finetuned_grid_thw = item['video_grid_thw'].to(base.device), item['video_grid_thw'].to(finetune.device)
        # base_input_ids, finetuned_input_ids = item['input_ids'].to(base.device), item['input_ids'].to(finetune.device)
        # base_attention_mask, finetuned_attention_mask = item['attention_mask'].to(base.device), item['attention_mask'].to(finetune.device)
        base_outputs, finetuned_outputs = [], []
        with torch.inference_mode():
            # base_output = base.generate(input_ids=base_input_ids, attention_mask=base_attention_mask, video_grid_thw=base_grid_thw, max_new_tokens=256, pixel_values_videos=base_video_clip)
            # finetuned_output = finetune.generate(input_ids=finetuned_input_ids, attention_mask=finetuned_attention_mask, video_grid_thw=finetuned_grid_thw, max_new_tokens=256, pixel_values_videos=finetuned_video_clip)
            base_output = base.generate(**batch, max_new_tokens=256)
            base_output_trimmed = [
                output_ids[len(in_ids):] for in_ids, output_ids in zip(batch.input_ids, base_output)
            ]
            finetuned_output = finetune.generate(**finetuned_batch, max_new_tokens=256)
            finetuned_output_trimmed = [
                output_ids[len(in_ids): ] for in_ids, output_ids in zip(batch.input_ids, finetuned_output)
            ]
        base_generated_text = processor.batch_decode(base_output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
        finetuned_generated_text = processor.batch_decode(finetuned_output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
        base_outputs.append(base_generated_text)
        finetuned_outputs.append(finetuned_generated_text)
            
        # find action and reasoning and only take those
        base_output, base_action = majority_voting_action(base_outputs)
        finetuned_output, finetuned_action = majority_voting_action(finetuned_outputs)
        if base_action == ground_action:
            base_acc += 1
        if finetuned_action == ground_action:
            finetuned_acc += 1
        
        print(f"Iteration: {ind}, Base: {base_output}, Generated: {finetuned_output}")
        print(f"Iteration time {time.time() - start_time}")
        data.append([video_path, base_generated_text, finetuned_generated_text])
    
    print(f"Base acc: {base_acc / len(dataset)}, Finetuned acc: {finetuned_acc / len(dataset)}")

    with open('output_comparison.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == '__main__':
    args = parser.parse_args()
    print(torch.cuda.is_available())
    if 'qwen' in args.base_model.lower():
        processor = AutoProcessor.from_pretrained("./models/Qwen2-VL-7B-Instruct", use_fast=False)
    else:
        processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", use_fast=False)
    base_model = load_model(args.base_model)
    finetuned_model = load_model(args.finetuned)
    print(base_model.hf_device_map)
    print(finetuned_model.hf_device_map)
    
    # processor.tokenizer.pad_token = processor.tokenizer.bos_token
    # finetuned_model.config.pad_token_id = finetuned_model.config.bos_token_id
    processor.tokenizer.padding_side = "left"
    finetuned_model.padding_side = 'left'
    dataset = load_dataset(args.hf_dataset).with_format('torch')
    file_dir = 'good_data'
    test_dataset = dataset['test'].with_format('torch')
    run_inference(test_dataset, base_model, finetuned_model, processor)
# for inference, we might just have to clone the dataset regardless..., so upload both, then clone dataset when running inference
