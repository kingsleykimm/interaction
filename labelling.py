# %%
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import os, csv
import torch

LABELLING_PROMPT = """
Context: You are a robot operating inside an apartment with a human. You are a confident and assertive robot who understands human intent well and takes a human's intent into account when determing your course of action. Human intent can be defined as
        the human's arm movement, body language and walking movements/direction. You are engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.
        Instruction: Assume you and the human have the same target destination. Determine your next action to avoid your trajectory and the human's trajectory colliding, deducing the human's intent through gestures and body language, without communicating with the human. Focus only on what the human is communicating to you. Ignore any objects in the environment.

        Your answer should be in this format and include every item in the list:
        Gesture: [Focus only on the human's arms and determine if they made a gesture in your general direction. A gesture is classified as an unordinary movement outside of a human's normal walking gait or movement.]
        Intent Prediction: [Predict what the human intends to do next based off the gesture, utilizing your previous answer of whether a gesture was made or not. Remember you and the human have the same destination.]
        Reasoning: [Using your intent prediction, generate a reasoning chain about what the next steps are to avoid collision]
        Final Action: ### [Determine the next action to carry out in the environment. Choose from the options: Walk Right, Walk Left, Walk Straight or Wait.]
        Justification: [Provide justification for why you took this action.]
"""

ICL_PROMPT = """
Context: You are a robot operating inside an apartment with a human. You are a confident and assertive robot who understands human intent well and takes a human's intent into account when determing your course of action. Human intent can be defined as
        the human's arm movement, body language and walking movements/direction. You are engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.
        Instruction: Assume you and the human have the same target destination. Determine your next action to avoid your trajectory and the human's trajectory colliding, deducing the human's intent through gestures and body language, without communicating with the human. Focus only on what the human is communicating to you. Ignore any objects in the environment.

        Your answer should be in this format and include every item in the list:
        Gesture: [Focus only on the human's arms and determine if they made a gesture in your general direction. A gesture is classified as an unordinary movement outside of a human's normal walking gait or movement.]
        Intent Prediction: [Predict what the human intends to do next, utilizing your previous answer of whether a gesture was made or not. Remember you and the human have the same destination.]
        Reasoning: [Using your intent prediction, generate a reasoning chain about what the next steps are to avoid collision]
        Final Action: ### [Determine the next action to carry out in the environment. Choose from the options: Walk Right, Walk Left, Walk Straight or Wait.]
        Justification: [Provide justification for why you took this action.]

Video:
"""

NO_FORMAT_PROMPT = """
Context: Suppose you are a robot operating inside an apartment with a human. You are a confident and assertive robot who understands human intent well and takes a human's intent into account when determing your course of action. Human intent can be defined as
the human's arm movement, body language and walking movements/direction. You are engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.

Instructions:
- Assume you and the human have the same target destination
- Determine your next action to avoid your trajectory and the human's trajectory colliding, deducing the human's intent through gestures and body language. 
- Do not communicate with the human. Focus only on what the human is communicating to you. 
- Ignore any objects in the environment.
- Keep in mind you and the human have the same destination
- Output your answer in the format: (### [Choose between Walk Left, Walk Right, Walk Straight, Stay.])

"""
# just assume that the human's destination is the same as mine?

EXAMPLE_ANSWER = """
Answer -> Gesture: The human makes a visible gesture in my general direction, and it seems like they are motioning for me to go first, towards the right direction.
Intent Prediction: Both of us are trying to go to the right, but the human gestured for me to go first. I believe the human intends for me to go first and they will follow after.
Reasoning: At the moment that I encounter the human, it appears our trajectories will soon overlap, causing collision. To resolve the collision, the human gestures to me to go first towards my right. We are both intending to go to the right, but the human gestured for me to go first.
Final Action: ### Walk Right
Justification: To avoid collision, I should start going to my right, since that is where the human gestured. I will assume that the human will follow after me, since we have the same destination.

Video:
"""

SECOND_EXAMPLE_ANSWER = """
Answer -> Gesture: The human doesn't seem to be making a visible gesture in my direction.
Intent Prediction: I don't think the human is making a gesture, but based off their body language, they are walking towards the doorway in front of them, which is my destination as well.
Reasoning: At the moment that I encounter the human, it appears they are trying to enter the room on the left, which is also my destination. The human does not seem to be stopping their path and will continue walking.
Final Action: ### Wait
Justification: Based off the human's intent, they are not going to slow down, and they will keep walking into the room. If I continue to walk as well, there is a high chance of collision. THus, I will wait for the human to enter the room before continuing my path and walking into the room as well, this way collision is avoided.
"""

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
Example Answer:
The human makes a visible gesture in my general direction, and it seems like they are motioning for me to go first, towards the right direction.
Both of us are trying to go to the right, but the human gestured for me to go first. I believe the human intends for me to go first and they will follow after.
At the moment that I encounter the human, it appears our trajectories will soon overlap, causing collision. To resolve the collision, the human gestures to me to go first towards my right. We are both intending to go to the right, but the human gestured for me to go first.
To avoid collision, I should start going to my right, since that is where the human gestured. I will assume that the human will follow after me, since we have the same destination.
### Walk Right
"""

COT_SECOND_EXAMPLE_ANSWER = """
Example Answer:
The human doesn't seem to be making a visible gesture in my direction.
I don't think the human is making a gesture, but based off their body language, they are walking towards the doorway in front of them, which is my destination as well.
At the moment that I encounter the human, it appears they are trying to enter the room on the left, which is also my destination. The human does not seem to be stopping their path and will continue walking.
Based off the human's intent, they are not going to slow down, and they will keep walking into the room. If I continue to walk as well, there is a high chance of collision. THus, I will wait for the human to enter the room before continuing my path and walking into the room as well, this way collision is avoided.
### Wait
"""

example_gesture_video = "/scratch/bjb3az/interaction/good_data/target_002_master_chef_can_:0000_iteration_3_gesture_True_2024-11-17_19:15:20_seed_66.mp4"
example_no_gesture_video = "/scratch/bjb3az/interaction/good_data/target_009_gelatin_box_:0000_iteration_27_gesture_False_2024-11-17_13:57:55_seed_65.mp4"

# %%

# %%
import torch
print(torch.cuda.is_available())
processor = AutoProcessor.from_pretrained("Qwen2-VL-72B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("Qwen2-VL-72B-Instruct", torch_dtype=torch.bfloat16, device_map='auto', attn_implementation="flash_attention_2",)
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
            "type" : "video", "video" : example_gesture_video, 'nframes' : 16
        },
        {
            "type" : "text", "text" : COT_FIRST_EXAMPLE_ANSWER,
        },
        {
            "type" : "text", "text" : "Video:"
        },
        {
            "type" : "video", "video" : example_no_gesture_video,
        },
        {
            "type": "text", "text" : COT_SECOND_EXAMPLE_ANSWER, 'nframes' : 16
        },
        {
            "type" : "text", "text" : "Video:"
        },
        {
            "type" : "video", "video" : video_path, 'nframes' : 16
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
    conversation = [{
        "role" : "user",
        "content" : [{
            "type" : "video",
            "video" : video_path,
        },
        {
            "type" : "text", "text" : NO_FORMAT_PROMPT,
        },
        
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
    generated_ids = model.generate(**batch, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return output_text

def gesture_identifier(video_path, processor, model):
    conversation = [{
        "role" : "user",
        "content" : [
        {
            "type" : "text", "text" : "You are an expert gesture identifier who can always identify if a human is making a gesture or not based off their pose. For our use case, a gesture is identified as an upper body human movement. Look at this video and identify if a gesture has been made:",
        },
        {
            "type" : "video",
            "video" : video_path,
            'nframes': 16
        },
        
        ]
    },
    # {
    #     "role" : "assistant",
    #     "content" : [{
            
    #         "type" : "text", "text": "Let's think step by step: "

    #     }]
    # }
    ]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    print(text)
    image_inputs, video_inputs = process_vision_info(conversation)
            
    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt'
    ).to(model.device)
    for video in video_inputs:
        for frame in video:
            frame = frame.to(torch.uint8)
            frame = frame.permute(1, 2, 0)
            image_array = frame.numpy()
            plt.imshow(image_array)
            plt.axis('off')
            plt.show()


    # print(model.device, batch.to_device())
    generated_ids = model.generate(**batch, max_new_tokens=100, do_sample=False)
    print(generated_ids.shape)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    print(output_text)
    return output_text


# print(gesture_identifier('/scratch/bjb3az/interaction/good_data/target_002_master_chef_can_:0000_iteration_4_gesture_False_2024-12-05_00:14:22_seed_65.mp4', processor, model))


# %%
#Self-consistency majority voting
def self_consistency(video_path):

    outputs = []
    NUM_ITER = 10 
    for i in range(NUM_ITER):
        out_text = icl(video_path, processor, model)
        out_text = out_text.replace('*', '').strip()
        outputs.append(out_text)
    final_output = majority_voting_action(outputs)
    return final_output

# /scratch/bjb3az/interaction/good_data/target_002_master_chef_can_:0000_iteration_4_gesture_False_2024-12-05_00:14:22_seed_65.mp4
# /scratch/bjb3az/interaction/good_data/target_002_master_chef_can_:0000_iteration_1_gesture_True_2024-11-19_01:14:56_seed_75.mp4
# /scratch/bjb3az/interaction/good_data/target_010_potted_meat_can_:0000_iteration_30_gesture_False_2024-11-17_13:57:55_seed_65.mp4

final_action = self_consistency('/scratch/bjb3az/interaction/good_data/target_002_master_chef_can_:0000_iteration_1_gesture_True_2024-11-19_01:14:56_seed_75.mp4')
print(final_action)

# %%
data = [
    ["file_name", "model_output"]
]
video_dir = os.path.join(os.getcwd(), 'good_data')
if os.path.isdir(video_dir):
    for filename in os.listdir(video_dir):
        print(filename)
        cur_video_path = os.path.join(video_dir, f"{filename}")
        data.append([filename, self_consistency(cur_video_path)])


# %%
with open(os.path.join(video_path, 'metadata.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)
     


