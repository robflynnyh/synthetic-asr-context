import argparse
from tqdm import tqdm
from synctxasr.data import dataset_functions
from synctxasr.wer import word_error_rate_detail
from synctxasr.misc import int_or_none
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
import os
from functools import partial

def preprocess_function(examples, tokenizer):
    formatted_examples = []
    prompt_lengths = []

    # Process each example individually
    for i in range(len(examples["generation"])):
        # Format the prompt without the answer
        prompt_messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": f"Given the following output from my ASR model, predict a plausable sentence that could have come before it: {examples['generation'][i]}."}]}
        ]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=True, return_tensors=None)
        prompt_lengths.append(len(prompt_text))

        full_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": examples["previous_generation"][i]}]}
        ]

        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
        formatted_examples.append(full_text)

    # Tokenize the formatted examples
    tokenized_outputs = tokenizer(
        formatted_examples,
        padding=True,
        padding_side="right",
        truncation=False,
        return_tensors="pt",
    )
    tokenized_outputs["labels"] = tokenized_outputs["input_ids"].clone()
    for i, length in enumerate(prompt_lengths):
        tokenized_outputs["labels"][i][:length] = -100 
    
    return tokenized_outputs

def get_dataset(args, split='train'): # preload all text
    path = os.path.join(args.dataset_path, split)
    files = os.listdir(path)
    dataset = []
    for file_path in tqdm(files):
        full_file_path = os.path.join(path, file_path)
        file = torch.load(full_file_path)
        dataset.append({k:v for k,v in file.items() if k != 'waveform'}) # omit waveform

    return Dataset.from_list(dataset)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = get_dataset(args, split='train')
    train_tokenized = train_dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    print(train_tokenized)

          

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', type=str, default='/store/store4/data/huggingface_models/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_path', type=str, default='/store/store4/data/TEDLIUM3_Whisper_tiny_en_outputs/')



    args = parser.parse_args()    
    if not torch.cuda.is_available(): args.device = 'cpu'

    main(args)