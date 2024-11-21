# Load model directly
import os
import yaml
import datetime
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from transformers import Trainer
from argparse import ArgumentParser
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset


USE_LORA = False
MAX_LENGTH = 512
BATCH_SIZE = None
NUM_FRAMES = 32

CHECKPOINT_DIR = '/scratch/bjb3az/interaction/model_ckpt'

parser = ArgumentParser()
parser.add_argument('--dataset_path', type=str, help="name of the HF dataset to use")
parser.add_argument('--num_frames', type=int, help="Number of frames to split videos into")
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


class LlavaNextVideoDataCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        padded_inputs = self.processor.tokenizer.pad(
            {
                "input_ids": [feat['input_ids'][0] for feat in features], # each element is one batch only so we slice [0]
                "attention_mask": [feat['attention_mask'][0] for feat in features],
            },
            padding=True,
            return_tensors="pt",
        )

        labels = padded_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        padded_inputs["labels"] = labels
        padded_inputs["pixel_values_videos"] = torch.cat([feat['pixel_values_videos'] for feat in features], dim=0)

        return padded_inputs

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
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype= torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map="auto"
        )
    return model



if __name__ == "__main__":
    args = parser.parse_args()
    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", use_fast=False) # this is already the tokenizer
    processor.tokenizer.padding_side = "right"
    dataset = load_dataset(args.dataset_path)
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset['train'].with_format('torch')
    # for item in train_dataset:
    #     print(item)
    test_dataset = dataset['test'].with_format('torch')
    model = load_model(args.model_path)
    model.padding_side = 'right'
    trainer = Trainer(
        model = model,
        tokenizer = processor,
        data_collator=LlavaNextVideoDataCollatorWithPadding(processor=processor),
        train_dataset=train_dataset,
        eval_dataset=test_dataset, # need one
        args=load_training_args(args.config_path),

    )
    # torch.cuda.empty_cache()
    trainer.train()
    trainer.push_to_hub("End of training")
    # if we already have the pretrained model, we might as well jsut run inference on this 
    for example in test_dataset:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": MASTER_PROMPT},
                    {"type": "video"},
                    ],
            },
        ]

    # Set add_generation_prompt to add the "ASSISTANT: " at the end
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        batch = processor(
            text=prompt,
            videos=None,
            return_tensors="pt"
        ).to(model.device)
        output = model.generate(**batch, pixel_values_videos=example['pixel_values_videos'], max_length=512)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        print(generated_text)
