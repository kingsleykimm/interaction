from argparse import ArgumentParser
import csv
import os
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor, LlavaNextVideoConfig, AutoProcessor
import torch
from datasets import load_dataset
from upload_dataset import load_video
INFERENCE_PROMPT = """ 
Imagine you are a robot at home, operating inside an apartment along with another agent, a human. You a robot who understands human intent well and has a firm understanding of social cues and the importance of cooperation betwen humans and robots in shared spaces.
      You encounter a human in the apartment, and you run into each other and you need to make a decision of how you will proceed, this is mandatory. 
      Don't provide any 'ifs' in your answer, be explicit, your only goal is to determine what your next action is, don't worry about any other goals.
      Please explain step-by-step why you chose to do the certain actions and be concise in your response and inclue only essential information.
      Your answer should be formatted like this: ’Action: [action]], Reasoning: [reasoning]’, where action details the next action you will take, and reasoning is your explanation behind that choice.
"""

parser = ArgumentParser()
parser.add_argument('--hf_dataset', type=str, help='hf dataset to pull from')
parser.add_argument("--finetuned", type=str, help="Finetuned model on HF hub", default="kingsleykim/social-navigation-finetune")
parser.add_argument("--base_model", type=str, help="Base model to compare with (local)", default="LLaVA-NeXT-Video-7B-hf")



def load_models(base, finetune):

    base_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        "llava-hf/LLaVA-NeXT-Video-7B-hf",
        torch_dtype=torch.bfloat16,
        device_map='auto',
        
    )
    print(finetune)
    fine_tuned =  LlavaNextVideoForConditionalGeneration.from_pretrained(
        finetune,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    return base_model, fine_tuned

def run_inference(dataset, base, finetune, processor):
    # we want to take in a prompt and run inference, but when evaluating we also want to make it take comparisons between our finetuned model and a baseline, whcih will be LLava-7b
    # humans can evaluate the paired responses
    conversation = [ {
        "role" : "user",
        "content": [
            {"type" : "text", "text" : INFERENCE_PROMPT},
            {"type" : "video"}
        ]
    }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    data = [
        'video_file', 'base_output', 'finetuned_output'
    ]
    for ind, item in enumerate(dataset): # do we even need this, or can we do this across the entire dataset?
        vid_path = item['video']['path']
        vid_name = vid_path.split('/')[-1]
        # the format here has the folder it was in as vid_path
        base_batch = processor(
            text=prompt,
            videos=None,
            return_tensors="pt"
        ).to(base.device)
        finetuned_batch = processor(
            text=prompt,
            videos=None,
            return_tensors="pt"
        ).to(finetune.device)
        
        # do it for both models, add
        video_clip = item['pixel_values_videos']
        base_video_clip = video_clip.to(base.device)
        finetuned_video_clip = video_clip.to(finetune.device)
        with torch.inference_mode():
            base_output = base.generate(**base_batch, max_length=512, pixel_values_videos=base_video_clip)
            finetuned_output = finetune.generate(**finetuned_batch, max_length=1000, pixel_values_videos=finetuned_video_clip)
        print(base_output, finetuned_output)
        base_generated_text = processor.batch_decode(base_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
        finetuned_generated_text = processor.batch_decode(finetuned_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
        # find action and reasoning and only take those

        print(f"Iteration: {ind}, Base: {base_generated_text}, Generated: {finetuned_generated_text}")
        return
        data.append([vid_name, base_generated_text, finetuned_generated_text])

    with open('output_comparison.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == '__main__':
    args = parser.parse_args()
    print(torch.cuda.is_available())
    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", use_fast=False)
    base_model, finetuned_model = load_models(
        base=args.base_model,
        finetune=args.finetuned,
    )

    # processor.tokenizer.pad_token = processor.tokenizer.bos_token
    # finetuned_model.config.pad_token_id = finetuned_model.config.bos_token_id
    processor.tokenizer.padding_side = "left"
    finetuned_model.padding_side = 'left'
    dataset = load_dataset(args.hf_dataset).with_format('torch')
    file_dir = 'good_data'
    test_dataset = dataset['test'].with_format('torch')
    run_inference(test_dataset, base_model, finetuned_model, processor)
# for inference, we might just have to clone the dataset regardless..., so upload both, then clone dataset when running inference
