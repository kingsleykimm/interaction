from argparse import ArgumentParser
import os
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch
from datasets import load_dataset
from upload_dataset import load_video
INFERENCE_PROMPT = """ 
Imagine you are a robot at home, operating inside an apartment or household along with another agent, a human. You are a confident robot who knows
      how to operate in these environments.
      You are going to receive a video showing an interaction between you and the human, where you run into each other and need to make a decision.
      The human might perform some sort of action that signals you to go ahead, but regardless, you must give the next action you will take, this is mandatory. Focus only on the gestures/movement
      the human is making, not on anything else in the environment. 
      Please explain step-by-step why you chose to do the certain actions and be concise in your response but include any essential information.
"""

parser = ArgumentParser()
parser.add_argument('--hf_dataset', type=str, help='hf dataset to pull from')
parser.add_argument("--finetuned", type=str, help="Finetuned model on HF hub", default="kingsleykim/social-navigation-finetune")
parser.add_argument("--base_model", type=str, help="Base model to compare with (local)", default="LLaVA-NeXT-Video-7B-hf")



def load_models(base, finetune):
    base_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        base,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    fine_tuned =  LlavaNextVideoForConditionalGeneration.from_pretrained(
        finetune,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    return base_model, fine_tuned

def run_inference(dataset, model, processor, file_dir):
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
    for item in dataset: # do we even need this, or can we do this across the entire dataset?
        vid_path = item['video']['path']
        vid_name = vid_path.split('/')[-1]
        # the format here has the folder it was in as vid_path
        new_path = os.path.join(os.getcwd(), file_dir, vid_name)
        video = load_video(new_path, 32)
        batch = processor(
            text=prompt,
            videos=video,
            return_tensors="pt"
        )
        # do it for both models, add
        output = model.generate(**batch)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)

        # we need to prune this slightly, since right now it's going to be pointing to a former link
    # TODO: write to CSV


if __name__ == '__main__':
    args = parser.parse_args()
    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", use_fast=False)
    base_model, finetuned_model = load_models(
        base=args.base_model,
        finetune=args.finetuned
    )
    dataset = load_dataset(args.hf_dataset).with_format('torch')
    file_dir = args.hf_dataset.split('/')[1]
    test_dataset = dataset['test'].with_format('torch')
    run_inference(test_dataset, finetuned_model, processor, file_dir)
# for inference, we might just have to clone the dataset regardless..., so upload both, then clone dataset when running inference
