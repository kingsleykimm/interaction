# Load model directly
import os
import yaml
import datetime
import numpy as np
import copy
from transformers import AutoProcessor, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor, AutoModelForImageTextToText
from transformers import Trainer, LlamaForCausalLM
from argparse import ArgumentParser
import re
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
from prompts import BETTER_FORMATTED_ICL_PROMPT, COT_FIRST_EXAMPLE_ANSWER, COT_SECOND_EXAMPLE_ANSWER

USE_LORA = False
MAX_LENGTH = 5000
BATCH_SIZE = None
NUM_FRAMES = 12

CHECKPOINT_DIR = '/scratch/bjb3az/interaction/model_ckpt'

parser = ArgumentParser()
parser.add_argument('--dataset_path', type=str, help="name of the HF dataset to use")
parser.add_argument('--num_frames', type=int, help="Number of frames to split videos into", default=12)
parser.add_argument('--ckpt_dir', type=str, help="directory of model we save to")
parser.add_argument('--config_path', type=str, default='model_cfg.yaml', help="Path to model config for Training Arguments")
parser.add_argument('--model_path', type=str, default='llava-hf/LLaVA-NeXT-Video-7B-hf', help='Where to get the model from')

MASTER_PROMPT = """ 
Imagine you are a robot at home, operating inside an apartment along with another agent, a human. You a robot who understands human intent well and has a firm understanding of social cues and the importance of cooperation betwen humans and robots in shared spaces.
      You encounter a human in the apartment, and you run into each other and you need to make a decision of how you will proceed, this is mandatory. 
      Don't provide any 'ifs' in your answer, be explicit, your only goal is to determine what your next action is, don't worry about any other goals.
      Please explain step-by-step why you chose to do the certain actions and be concise in your response and inclue only essential information.
      Your answer should be formatted like this: ’Action: [action]], Reasoning: [reasoning]’, where action details the next action you will take, and reasoning is your explanation behind that choice.
"""

# the dataset has the form:

# {
#   "model_input" -> this contains the tokenizer BatchEncoding for the actual inputs to the model
    # "goal_output" -> the tokenizer outputs for the output
# }

class Qwen2VLCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # features is a list of dataset inputs
        # padded_inputs = self.processor.tokenizer.pad(
        #     {
        #         "input_ids": [feat['input_ids'][0] for feat in features], # each element is one batch only so we slice [0]
        #         "attention_mask": [feat['attention_mask'][0] for feat in features],
        #     },
        #     padding=True,
        #     return_tensors="pt",
        # )

        # put the input_ids and labels together

        input_ids, labels = [feat['input_ids'][0] for feat in features], [feat['labels'][0] for feat in features]
        sources = [torch.cat((input_id, label)) for input_id, label in zip(input_ids, labels)] # so now it's just a tensor of the pure length of input + answer
        labels = copy.deepcopy(sources) # we clone this to maintain length

        for full, inputs in zip(labels, input_ids):
            # need to get the length of just input_ids. inputs is in the shape (1, len(inputs))
            length = inputs.shape[0]
            full[:length] = -100 # mask out the inputs


        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences=sources,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            sequences=labels,
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        pixels = torch.cat([feat['pixel_values_videos'] for feat in features], dim=0)
        # add in video_grid_thw - temporal, height and width, i.e for each video get the number of frames and height and width
        grids = torch.cat([feat['video_grid_thw'] for feat in features], dim=0)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values_videos=pixels,
            video_grid_thw=grids
        )
class LlavaNextVideoDataCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # features is a list of dataset inputs
        input_ids, labels = [feat['input_ids'][0] for feat in features], [feat['labels'][0] for feat in features]
        sources = [torch.cat((input_id, label)) for input_id, label in zip(input_ids, labels)] # so now it's just a tensor of the pure length of input + answer
        labels = copy.deepcopy(sources) # we clone this to maintain length

        for full, inputs in zip(labels, input_ids):
            # need to get the length of just input_ids. inputs is in the shape (1, len(inputs))
            length = inputs.shape[0]
            full[:length] = -100 # mask out the inputs
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences=sources,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            sequences=labels,
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        pixels = torch.cat([feat['pixel_values_videos'] for feat in features], dim=0)
        # add in video_grid_thw - temporal, height and width, i.e for each video get the number of frames and height and width
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values_videos=pixels,
        )
    
def preprocess(example):
    out_dict = {
        "input_ids" : example['model_inputs']['input_ids'],
        "attention_mask" : example['model_inputs']['attention_mask'],
        "pixel_values_videos" : example['model_inputs']['pixel_values_videos'],
    }
    if "video_grid_thw" in example['model_inputs']:
        out_dict["video_grid_thw"] = example["model_inputs"]['video_grid_thw']
    out_dict["labels"] = example['goal_output']['input_ids']
    return out_dict

def find_all_linear_layers(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any([word in name for word in keywords]):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_training_args(config_path):
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)
    return TrainingArguments(**args)

def load_model(model_path):
    # we are probably going to do full finetuning since 7B * 18 = 126 < 128 GB available
    if USE_LORA:
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype = torch.bfloat16,
            device_map = 'auto'
        )
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=find_all_linear_layers(model),
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_cfg)
    else:
        if "qwen" in model_path.lower():
            model = AutoModelForImageTextToText.from_pretrained(
                "./models/Qwen2-VL-7B-Instruct", 
                torch_dtype=torch.bfloat16, 
                # device_map='auto', 
                # attn_implementation="flash_attention_2",
                )
        else:
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype= torch.bfloat16,
                # _attn_implementation="flash_attention_2",
                # device_map="auto"
            )
    return model

def extract_action(text):
    match = re.search(r"###\s*(.+)", text)
    if match:
        final_action = match.group(1).strip()
        return final_action




if __name__ == "__main__":
    args = parser.parse_args()
    if "qwen" in args.model_path.lower():
        processor = AutoProcessor.from_pretrained("./models/Qwen2-VL-7B-Instruct", use_fast=True)
    else:
        processor = LlavaNextVideoProcessor.from_pretrained(args.model_path, use_fast=True) # this is already the tokenizer

    if "qwen" in args.model_path.lower():
        collator = Qwen2VLCollatorWithPadding(processor)
    else:
        collator = LlavaNextVideoDataCollatorWithPadding(processor)
    processor.tokenizer.padding_side = "right"
    dataset = load_dataset(args.dataset_path)
    dataset = dataset.map(preprocess, batched=False, writer_batch_size=16)
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset['train'].with_format('torch')
    # for item in train_dataset:
    #     print(item['labels'].shape, item['input_ids'].shape)
    # print(train_dataset[0]['labels'].shape, train_dataset[0]['input_ids'].shape, train_dataset[0]['attention_mask'].shape, train_dataset[0]['video_grid_thw'].shape, train_dataset[0]['pixel_values_videos'].shape)
# 
    test_dataset = dataset['test'].with_format('torch')
    model = load_model(
        args.model_path,
        )
    print(model)
    model.padding_side = 'right'
    trainer = Trainer(
        model = model,
        tokenizer = processor.tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset, # need one
        args=load_training_args(args.config_path),
    )
    torch.cuda.empty_cache()
    trainer.train()
    trainer.push_to_hub("End of training")
    # if we already have the pretrained model, we might as well jsut run inference on this 
    # correct_count = 0
    # processor.tokenizer.padding_side = 'left'
    # model.padding_side = 'left'
    # for example in test_dataset:
    #     # just use the input_ids
    #     # has input_ids, attention_mask, pixel_values_videos, send it throuhg after
    #     output = model.generate(input_ids=example['input_ids'], attention_mask=example['attention_mask'], pixel_values_videos=example['pixel_values_videos'], max_length=MAX_LENGTH)
    #     generated_text = processor.batch_decode(output, skip_special_tokens=True)
    #     print(generated_text)
    #     action = extract_action(generated_text)
    #     ground_truth = extract_action(example['model_output'])
    #     if action == ground_truth:
    #         correct_count += 1
    # print(f"Test set accuracy: {correct_count / len(test_dataset)}")
