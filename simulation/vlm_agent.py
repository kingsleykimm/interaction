from transformers import AutoProcessor, LlavaNextVideoProcessor, AutoModelForImageTextToText, LlavaNextVideoForConditionalGeneration
import torch
from habitat_sim.utils import viz_utils
import numpy as np
from .prompts import BETTER_FORMATED_ICL_PROMPT
import random, re
from collections import defaultdict


class VLMAgent:
    """
    At the moment of human sighting, the current observations up to now should be passed to the agent
    """
    def __init__(self, model_path, config):

        self.config = config
        self.load_items(model_path)
        self.last_num_frames = config.last_num_frames
        
        self.formatted_text = self.load_conversation()
    def load_items(self, model_path):
        if "qwen" in model_path.lower():
            self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                device_map='auto', 
                attn_implementation="flash_attention_2",)
        else:
            self.processor = LlavaNextVideoProcessor.from_pretrained(model_path, use_fast=True) # this is already the tokenizer
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype= torch.bfloat16,
                _attn_implementation="flash_attention_2",
                device_map="auto"
            )
    def load_conversation(self):
        self.conversation = conversation = [{
            "role" : "user",
            "content": [
                {"type" : "text", "text" : BETTER_FORMATTED_ICL_PROMPT},
                {"type" : "text", "text" : "Video:"},
                {"type" : "video", "video" : "placeholder"}
            ]
            },
            {
                "role" : "assistant",
                "content" : [{
                    
                    "type" : "text", "text": "Let's think step by step: "

                }]
            }
            ]
        return self.processor.apply_chat_template(conversation, tokenize=False, continue_final_message=True)
    def infer_action(self, habitat_obs):
        # compress and turn it into video
        # observations are in 400 x 400 -> convert to VLM format
        # take in the last 12 frames
        if last_frames.shape[0] > self.last_num_frames:
            last_frames = habitat_obs[-self.last_num_frames:]
        if isinstance(last_frames, np.ndarray):
            last_frames = torch.from_numpy(last_frames)
        last_frames = last_frames.float()
        last_frames = torch.unsqueeze(last_frames, 0)
        single_batch = self.processor(
            [self.formatted_text],
            videos=last_frames,
            return_tensors='pt'
        )
        output_ids = self.model.generate(**single_batch)
        output_trim = [outputs[len(inputs):] for inputs, outputs in enumerate(single_batch.input_ids, output_ids)]
        output_text = self.processor.batch_decode(output_trim, skip_special_tokens=True, clean_up_tokenization_space=True)[0].strip()
        action = self.majority_voting_action([output_text])

        # some things to consider:
        # how many milliseconds / environment steps do we want the robot to take?
        # for wait it's easy
        # for walk, do we continue on our current course of action?
        if action == "Walk Left":
        
        elif action == "Walk Right":
            # find the best coord to go
        else:
            return "Walk Straight"
            # Walk Straight / continue on course for the next 10 seconds


        # these are already in the format of (num_frames, num_channels, H, W), since by default it gets the RGB, it's just in numpy form

    def majority_voting_action(self, outputs):
        if len(outputs) == 1:
            match = re.search(r"###\s*(.+)", outputs[0])
            if match:
                final_action = match.group(1).strip()
                return final_action
            else:
                print("No match found.")
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


"""
Best API / actions to use

say(message)
ask(question, options)
predict_human_path() -> predict_human_path argument
get_current_location()
walk(direction, distance)
find_next_location() -> uses walk
get_current_room()
does_collision_exist(human_walk_vector, current_position)
gesture_performed() -> bool

In most scenarios, the VLM should identify the human's walk and predict where they are going, think about if a collision exists with the current vector,
if it does, then find a new walk to take and how far it should go

How do we incorporate gestures into this? IF a gesture is done, incorporate it into the predict_human_path().
We can test both pipelines, using Qwen VL 72B as everything and make it output generate code. The benefit is that it gets to incorporate the visual information. Is there a finetuning chance here?
How could we classify a trajectory for RLHF? make distance throught trajectory as the reward? higher distance (normalized by the average distance in trajectory = more reward) (normalize to 0-1)
Or use VLM output -> starcoder, we lose the visual context though
"""