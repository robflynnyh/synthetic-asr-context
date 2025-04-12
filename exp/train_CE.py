import argparse
from tqdm import tqdm
from synctxasr.data import dataset_functions
from synctxasr.wer import word_error_rate_detail
from synctxasr.misc import int_or_none
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from accelerate import Accelerator

from datasets import Dataset
import torch
import os
from functools import partial
import wandb

def test_output(model, tokenizer):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an helpful assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Given the following output from my ASR model, predict a plausable sentence that could have come before it: buti was so utterly unqualified for this project and so utterly ridic and ignored the brief.."}]
        },
    ]


    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id    
        )
        

    print(outputs.shape )
    outputs = tokenizer.batch_decode(outputs)
    response = outputs[0].split("<start_of_turn>model\n")[-1].strip().split("<end_of_turn>")[0].strip()

    print("Model's generated response:", response)

class DataCollatorWithPaddingAndLabels(DataCollatorWithPadding): # pad the labels aswell as the input, padded from the left as this is default for the parent
    def __call__(self, features):
    
        labels = [f["labels"] for f in features]
        for f in features:
            del f["labels"]

        batch = super().__call__(features)
        label_max_length = max([len(l) for l in labels])
        batch["labels"] = []
        for i, f in enumerate(features):
            label = labels[i]
            if len(label) < label_max_length:
                label = [-100] * (label_max_length - len(label)) + label
            batch["labels"].append(label)
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long).to(batch["input_ids"].device)

        return batch

def preprocess_function(examples, tokenizer):
    formatted_examples = []
    prompt_lengths = []
    # Process each example individually
    for i in range(len(examples["generation"])):
        # Format the prompt without the answer
        prompt_messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": f"Given the following output from the ASR model, please predict a plausable sentence that could have come before it: '{examples['generation'][i]}'."}]},
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
        padding=False,
        truncation=False,
        return_tensors=None,
    )
    tokenized_outputs["labels"] = []
    for i, length in enumerate(prompt_lengths):
        tokenized_outputs["labels"].append([-100] * length + tokenized_outputs["input_ids"][i][length:])    
    
    return tokenized_outputs


def get_dataset(args, split='train', max_files=10000): # preload all text
    path = os.path.join(args.dataset_path, split)
    files = os.listdir(path)
    dataset = []
    for i, file_path in enumerate(tqdm(files)):
        if max_files > 0 and i > max_files: break # for debugging purposes
        full_file_path = os.path.join(path, file_path)
        file = torch.load(full_file_path)
        if file['previous_generation'] == None: continue

        file['previous_generation'] = file['previous_generation'].strip()
        file['previous_gold_text'] = file['previous_gold_text'].strip()

        dataset.append({k:v for k,v in file.items() if k != 'waveform'}) # omit waveform

    return Dataset.from_list(dataset)

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    accelerator = Accelerator()
    
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = get_dataset(args, split='train')
    dev_dataset = get_dataset(args, split='dev')

    train_tokenized = train_dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    dev_tokenized = dev_dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing dev dataset",
    )


    model = AutoModelForCausalLM.from_pretrained(args.language_model)

    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorWithPaddingAndLabels(tokenizer=tokenizer, padding='longest', return_tensors='pt')
 
    if not args.no_wandb and accelerator.is_local_main_process: wandb.init(project="synctxasr")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=15,
        save_strategy="steps",
        load_best_model_at_end=True,
        warmup_steps=1000,
        bf16=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=4,
        report_to=["wandb"] if not args.no_wandb else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=4,
        optim="adafactor",
        #fsdp="full_shard",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)],
    )

    trainer.evaluate()
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final_model"))

    if not args.no_wandb and accelerator.is_local_main_process:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', type=str, default='/store/store4/data/huggingface_models/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_path', type=str, default='/store/store4/data/TEDLIUM3_Whisper_tiny_en_outputs/')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-8)
    parser.add_argument('--output_dir', type=str, default='/store/store5/data/acp21rjf_checkpoints/synctxasr/')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=0)

    args = parser.parse_args()    
    if not torch.cuda.is_available(): args.device = 'cpu'

    main(args)