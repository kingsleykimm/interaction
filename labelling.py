# %%
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import os, csv
import torch
from prompts import *



# %%

# %%
import torch
print(torch.cuda.is_available())
processor = AutoProcessor.from_pretrained("./models/Qwen2-VL-72B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("./models/Qwen2-VL-72B-Instruct", torch_dtype=torch.bfloat16, device_map='auto', attn_implementation="flash_attention_2",)
print(model.device)



# %%
# majority voting
import re
import random
from collections import defaultdict
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
        return "NO ACTION"
    fin_action = random.sample(actions_sample, 1)[0]

    final_outputs = []
    for text in outputs:
        match = re.search(r"###\s*(.+)", text)
        if match:
            final_action = match.group(1).strip()
            if final_action == fin_action:
                final_outputs.append(text)
    if len(final_outputs) == 0:
        return "NO ACTION"
    return random.sample(final_outputs, 1)[0]




# %%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def label_one(video_path, processor, model):
    conversation = [{
        "role" : "user",
        "content" : [{
            "type" : "video",
            "video" : video_path,
        },
        {
            "type" : "text", "text" : LABELLING_PROMPT,
        }]
    }
    ]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt'
    ).to(model.device)
    # print(model.device, batch.to_device())
    generated_ids = model.generate(**batch, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    print(output_text)
    return output_text

def icl(video_path, processor, model):
    conversation = [{
        "role" : "user",
        "content" : [
        {
            "type" : "text", "text" : BETTER_FORMATTED_ICL_PROMPT,
        },
        {
            "type" : "text", "text" : "Video:"
        },
        {
            "type" : "video", "video" : example_gesture_video, 'nframes' : 12
        },
        {
            "type" : "text", "text" : COT_FIRST_EXAMPLE_ANSWER,
        },
        {
            "type" : "video", "video" : example_no_gesture_video,
        },
        {
            "type": "text", "text" : COT_SECOND_EXAMPLE_ANSWER, 'nframes' : 12
        },
        {
            "type" : "video", "video" : video_path, 'nframes' : 12
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
    image_inputs, video_inputs = process_vision_info(conversation)
    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt'
    ).to(model.device)
    # print(model.device, batch.to_device())
    input_ids = batch.input_ids
    attention_mask = batch.attention_mask
    generated_ids = model.generate(**batch, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return output_text

def no_format_output(video_path, processor, model):
    conversation = [
        {
            "role" : "system",
            "content": [
                {"type" : "text", "text" : NO_FORMAT_PROMPT},
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

    text = processor.apply_chat_template(conversation, tokenize=False, continue_final_message=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt'
    ).to(model.device)
    # print(model.device, batch.to_device())
    generated_ids = model.generate(**batch, max_new_tokens=184)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return output_text

def gesture_identifier(video_path, processor, model):
    conversation = [
        {
            "role" : "system",
            "content": [
                {"type" : "text", "text" : CODE_PROMPT},
            ]
        },
            
        {
            "role" : "user",
            "content": [
                {"type" : "video", "video" : example_gesture_video}
            ]
        },
        {
            "role" : "assistant",
            "content" : [{
                
                "type" : "text", "text": ASSISTANT_RESPONSE_1

            }]
        },
        {
            "role" : "user",
            "content" : [
                {"type" : "video", "video" : example_no_gesture_video}
            ]
        },
        {
            "role" : "assistant",
            "content" : [
                {"type" : "text" , "text" : ASSISTANT_RESPONSE_2}
            ]
        },
        {
            "role" : "user",
            "content" : [{
                "type" : "video", "video": video_path
            }]
        }

        ]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
            
    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt'
    ).to(model.device)
    # for video in video_inputs:
    #     for frame in video:
    #         frame = frame.to(torch.uint8)
    #         frame = frame.permute(1, 2, 0)
    #         image_array = frame.numpy()
    #         plt.imshow(image_array)
    #         plt.axis('off')
    #         plt.show()


    # print(model.device, batch.to_device())
    generated_ids = model.generate(**batch, max_new_tokens=102)
    print(generated_ids.shape)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return output_text


# print(gesture_identifier('/scratch/bjb3az/interaction/good_data/target_002_master_chef_can_:0000_iteration_4_gesture_False_2024-12-05_00:14:22_seed_65.mp4', processor, model))


# %%
#Self-consistency majority voting
def self_consistency(video_path):

    outputs = []
    NUM_ITER = 1
    for i in range(NUM_ITER):
        out_text = no_format_output(video_path, processor, model)
        out_text = out_text.replace('*', '').strip()
        outputs.append(out_text)
    final_output = majority_voting_action(outputs)
    return final_output

# /scratch/bjb3az/interaction/good_data/target_002_master_chef_can_:0000_iteration_4_gesture_False_2024-12-05_00:14:22_seed_65.mp4
# /scratch/bjb3az/interaction/good_data/target_002_master_chef_can_:0000_iteration_1_gesture_True_2024-11-19_01:14:56_seed_75.mp4
# /scratch/bjb3az/interaction/good_data/target_010_potted_meat_can_:0000_iteration_30_gesture_False_2024-11-17_13:57:55_seed_65.mp4

# final_action = self_consistency('/scratch/bjb3az/interaction/good_data/target_002_master_chef_can_:0000_iteration_1_gesture_True_2024-11-19_01:14:56_seed_75.mp4')
# print(final_action)

# %%
data = [
    ["file_name", "model_output"]
]
import time
video_dir = os.path.join(os.getcwd(), 'examples')
if os.path.isdir(video_dir):
    for ind, filename in enumerate(os.listdir(video_dir)):
        print(filename)
        # take out the example videos in the ICL
        from_root_fp = os.path.join(video_dir, filename)
        if from_root_fp == example_gesture_video or from_root_fp == example_no_gesture_video or "mp4" not in from_root_fp:
            continue
        start_time = time.time()
        cur_video_path = os.path.join(video_dir, f"{filename}")
        output = self_consistency(cur_video_path)
        print(output)
        data.append([filename, output])
        print(f"Iteration number {ind}, duration time {time.time() - start_time}")

# %%
with open(os.path.join(video_dir, 'metadata.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)
     


