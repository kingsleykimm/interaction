# Load model directly
import os
import yaml
import datetime
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from transformers import Trainer, TrainingArguments, HfArgumentParser
from argparse import ArgumentParser
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Video
from decord import VideoReader, cpu


USE_LORA = False
MAX_LENGTH = 512
BATCH_SIZE = None
NUM_FRAMES = 32

CHECKPOINT_DIR = '/scratch/bjb3az/interaction/model_ckpt'

parser = ArgumentParser()
parser.add_argument('--dataset_path', type=str, help="path/link to the hf or local datset")
parser.add_argument('--num_frames', type=int, help="Number of frames to split videos into")
parser.add_argument('--ckpt_dir', type=str, help="directory of model we save to")
parser.add_argument('--hf', action="store_true", help="Whether or not to use huggingface dataset")
parser.add_argument('--config_path', type=str, default='model_cfg.yaml', help="Path to model config for Training Arguments")
parser.add_argument('--model_path', type=str, default='LLaVA-NeXT-Video-7B-hf', help='Where to get the model from')

MASTER_PROMPT = """ 
Imagine you are a robot at home, operating inside an apartment or household along with another agent, a human. You are a confident robot who knows
      how to operate in these environments.
      You are going to receive a video showing an interaction between you and the human, where you run into each other and need to make a decision.
      The human might perform some sort of action that signals you to go ahead, but regardless, you must give the next action you will take, this is mandatory. Focus only on the gestures/movement
      the human is making, not on anything else in the environment. 
      Please explain step-by-step why you chose to do the certain actions and be concise in your response but include any essential information.
"""


def load_in_dataset(hf : bool, file_path, processor, num_frames):
    dataset = None
    if hf: # if loading from huggingface
        dataset = load_dataset(file_path)
    else: # local load
        dataset = load_dataset("videofolder", data_dir=file_path, split="train")
    dataset = dataset.cast_column("video", Video(decode=False)) # to remove the automatic decord
    def collate_fn(example):
    # data format will just have filename and label
        label = example["model_output"]
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
            return_tensors="pt"
        )
        return batch

    dataset = dataset.map(collate_fn, batched=False, num_proc=1,writer_batch_size=400)
    return dataset['train']
    
def load_video(video_path,num_frames):
    vr = VideoReader(uri=video_path, ctx=cpu(0)) # you need to install from source to use gpu ctx
    indices = np.arange(0, len(vr), len(vr) / num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames


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
            torch_dtype = torch.float16,
            device_map = 'auto'
        )
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=find_all_linear_layers(model),
            init_lora_weights="guassian",
        )
        model = get_peft_model(model, lora_cfg)
    else:
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype= torch.float16,
            _attn_implementation="flash_attention_2",
            device_map="auto"
        )
    return model



if __name__ == "__main__":
    args = parser.parse_args()
    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", use_fast=False) # this is already the tokenizer
    processor.padding_side = "right"
    dataset = load_in_dataset(True, args.dataset_path, processor, args.num_frames)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset['train'].with_format('torch')
    test_dataset = dataset['test'].with_format('torch')
    model = load_model(args.model_path)
    trainer = Trainer(
        model = model,
        tokenizer = processor,
        data_collator=LlavaNextVideoDataCollatorWithPadding(processor=processor),
        train_dataset=train_dataset,
        eval_dataset=test_dataset, # need one
        args=load_training_args(args.config_path)
    )
    # torch.cuda.empty_cache()
    print(model)
    trainer.train()
    trainer.model.push_to_hub('kingsleykim/social-navigation-finetune')

    


    # dataset = dataset.train_test_split(test_size=0.2)  don't put that in yet
