# %%
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import os, csv
import torch

LABELLING_PROMPT = """
Context: You are a robot operating inside an apartment with a human. You are a confident and assertive robot who understands human intent well and takes a human's intent into account when determing your course of action. Human intent can be defined as
        the human's arm movement, body language and walking movements/direction. You are engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.
        Instruction: Determine your next action to avoid your trajectory and the human's trajectory colliding, deducing the human's intent through gestures and body language, without communicating with the human. Focus only on what the human is communicating to you. Ignore any objects in the environment.

        Your answer should be in this format and include every item in the list:
        Gesture: [Focus only on the human's arms and determine if they made a gesture in your general direction. A gesture is classified as an unordinary movement outside of a human's normal walking gait or movement.]
        Intent Prediction: [Predict what the human intends to do next based off the gesture.]
        Reasoning: [Using your intent prediction, generate a reasoning chain about what the next steps are to avoid collision]
        Final Action: [Determine the next action to carry out in the environment]
        Justification: [Provide justification for why you took this action.]
"""

ICL_PROMPT = """
Context: You are a robot operating inside an apartment with a human. You are a confident and assertive robot who understands human intent well and takes a human's intent into account when determing your course of action. Human intent can be defined as
        the human's arm movement, body language and walking movements/direction. You are engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.
        Instruction: Determine your next action to avoid your trajectory and the human's trajectory colliding, deducing the human's intent through gestures and body language, without communicating with the human. Focus only on what the human is communicating to you. Ignore any objects in the environment.

        Your answer should be in this format and include every item in the list:
        Gesture: [Focus only on the human's arms and determine if they made a gesture in your general direction. A gesture is classified as an unordinary movement outside of a human's normal walking gait or movement.]
        Intent Prediction: [Predict what the human intends to do next based off the gesture.]
        Reasoning: [Using your intent prediction, generate a reasoning chain about what the next steps are to avoid collision]
        Final Action: [Determine the next action to carry out in the environment]
        Justification: [Provide justification for why you took this action.]

As an example, I will give you two example videos and provide example answers
After looking at the example, you will receive the actual video and provide your answers.
Example Videos:
"""

EXAMPLE_ANSWER = """
Answer for first video:
Gesture: The human makes a visible gesture in my general direction, and it seems like they are motioning for me to go first, towards the right direction.
Intent Prediction: Based off the gesture, I believe the human intends for me to go first, towards my right and they will follow after.
Reasoning: At the moment that I encounter the human, it appears our trajectories will soon overlap, causing collision. To resolve the collision, the human gestures to me to go first towards my right. This is most likely because the human plans on taking a different path from where they motioned to avoid collision.
Final Action: I will start walking towards the right
Justification: To avoid collision, I should start going to my right, since that is where the human gestured. I will assume that the human is planning on going a different direction to me

Answer for second video:
Gesture: The human doesn't seem to be making a visible gesture in my direction.
Intent Prediction: I don't think the human is making a gesture, but based off their body language, they are walking towards the doorway in front of them.
Reasoning: At the moment that I encounter the human, it appears we are both trying to enter the room on the left. The human does not seem to be stopping their path and will continue walking.
Final Action: I will wait and let the human go into the room first
Justification: Based off the human's intent, they are not going to slow down, and they will keep walking into the room. If I continue to walk as well, there is a high chance of collision. THus, I will wait for the human to enter the room before continuing my path and walking into the room as well, this way collision is avoided.
"""

example_gesture_video = "/scratch/bjb3az/interaction/good_data/target_002_master_chef_can_:0000_iteration_3_gesture_True_2024-11-17_19:15:20_seed_66.mp4"
example_no_gesture_video = "/scratch/bjb3az/interaction/good_data/target_009_gelatin_box_:0000_iteration_27_gesture_False_2024-11-17_13:57:55_seed_65.mp4"

# %%


# %%
print(torch.cuda.is_available())
processor = AutoProcessor.from_pretrained("Qwen2-VL-72B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("Qwen2-VL-72B-Instruct", torch_dtype=torch.bfloat16, device_map='auto')
print(model.device)



# %%
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
    generated_ids = model.generate(**batch, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text

def icl(video_path, processor, model):
    conversation = [{
        "role" : "user",
        "content" : [
        {
            "type" : "text", "text" : ICL_PROMPT,
        },
        {
            "type" : "video", "video" : example_gesture_video,
        },
        {
            "type" : "video", "video" : example_no_gesture_video,
        },
        {
            "type" : "text", "text" : EXAMPLE_ANSWER,
        },
        {
            "type" : "video", "video" : video_path
        }
        ]
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
    input_ids = batch.input_ids
    attention_mask = batch.attention_mask
    generated_ids = model.generate(**batch, max_new_tokens=256, do_sample=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    print(output_text)
    return output_text


icl("/scratch/bjb3az/interaction/good_data/target_009_gelatin_box_:0000_iteration_30_gesture_True_2024-11-15_12:25:53.mp4", processor, model)

# %%
data = [
    ["file_name", "model_output"]
]
video_dir = os.path.join(os.getcwd(), 'good_data')
if os.path.isdir(video_dir):
    for filename in os.listdir(video_dir):
        print(filename)
        cur_video_path = os.path.join(video_dir, f"{filename}")
        data.append([filename, icl(cur_video_path, processor, model)])


# %%
with open(os.path.join(video_dir, 'metadata.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)
     


