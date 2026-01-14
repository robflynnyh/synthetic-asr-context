import os
# import nvidia.cublas.lib
# import nvidia.cudnn.lib

# lib_path = (
#     os.path.dirname(nvidia.cublas.lib.__file__)
#     + ":" +
#     os.path.dirname(nvidia.cudnn.lib.__file__)
# )

# # Update LD_LIBRARY_PATH
# os.environ["LD_LIBRARY_PATH"] = lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
# print(os.environ["LD_LIBRARY_PATH"])

import argparse
from tqdm import tqdm
from synctxasr.data import dataset_functions
from synctxasr.wer import word_error_rate_detail
from synctxasr.misc import int_or_none
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator
import whisper
from datasets import Dataset
import torch
import os
from functools import partial
import wandb
from synctxasr.wer import word_error_rate_detail    
from faster_whisper import WhisperModel
from torch import nn
from typing import Optional, Tuple, Union

from transformers.models.whisper import modeling_whisper
class WhisperForConditionalGeneration(modeling_whisper.WhisperGenerationMixin, modeling_whisper.WhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: modeling_whisper.WhisperConfig):
        super().__init__(config)
        self.model = modeling_whisper.WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_target_positions = config.max_target_positions

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    @modeling_whisper.add_start_docstrings_to_model_forward(modeling_whisper.WHISPER_INPUTS_DOCSTRING)
    @modeling_whisper.replace_return_docstrings(output_type=modeling_whisper.Seq2SeqLMOutput, config_class=modeling_whisper._CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[modeling_whisper.EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], modeling_whisper.Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`. `sequence_length` should be smaller than or equal to `config.max_target_positions`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = modeling_whisper.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-100, reduction='none')
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            # move labels to correct device to enable PP
            loss = loss_fct(lm_logits.transpose(1, 2), labels)
      

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return modeling_whisper.Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

modeling_whisper.WhisperForConditionalGeneration = WhisperForConditionalGeneration
from synctxasr.asr import get_whisper




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_paths):
        self.tokenizer = tokenizer
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def format_example(example):
        # prompt_messages = [
        #     {"role": "system", "content": [{"type": "text", "text": "You are an helpful assistant. The user will supply you with a sentence that was generated by an ASR model. Your task is to predict a plausible sentence that could have come before it. Do not include any other information in your response."}]},
        #     {"role": "user", "content": [{"type": "text", "text": f"'{example}'."}]},
        # ] 
        prompt_messages = [
            {"role": "system", "content": "You are a helpful assistant. The user will supply you with a sentence that was generated by an ASR model. Your task is to predict a plausible sentence that could have come before it. Do not include any other information in your response."},
            {"role": "user", "content": f"'{example}'."},
        ]
        return prompt_messages
    
    @staticmethod
    def log_file_error(file_path):
        print(f'Logging error for file: {file_path}')
        with open('error_log.txt', 'a') as f:
            f.write(f"{file_path}\n")


    def __getitem__(self, idx):
        try:
            data = torch.load(self.file_paths[idx])
        except EOFError as e: 
            # want to prevent a single file corruption from crashing the entire training run
            # so we catch the error pick a new random file and log the responsible file path so it can be deleted or fixed 
            print(f"Error loading file {self.file_paths[idx]}: {e}")
            self.log_file_error(self.file_paths[idx])
            idx = torch.randint(0, len(self.file_paths), (1,)).item()
            return self.__getitem__(idx)
        
        audio = data['waveform'].numpy()
        whisper_generation = data['generation'] 
        gold_text = data['text']
        example = self.format_example(whisper_generation)
        original_wer = word_error_rate_detail([whisper_generation], [gold_text], use_cer=False, normalize=True)[0]

        return {
            'prompt': example,
            'answer': gold_text,
            'audio': audio,
            'original_wer': original_wer,
        }

def get_text_normal(model, sample, prompt):
    result = model.transcribe(
        audio=sample, 
        initial_prompt=prompt, 
        without_timestamps=True, 
        language='en', 
        task='transcribe',
        beam_size=5,
    ) 
    return result['text'].strip()

def get_text_faster_whisper(model, sample, prompt):
    result = model.transcribe(
        audio=sample, 
        initial_prompt=prompt, 
        without_timestamps=True, 
        language='en', 
        task='transcribe',
        beam_size=5,
    )[0]
    text = " ".join([el.text for el in result]).strip()
    return text 



# Dummy reward function: count the number of unique characters in the completions
# def reward_func(asr_model):
#     def reward_num_unique_chars(prompts, answer, completions, audio, original_wer, **kwargs):
#         text_completions = [el[0]['content'] for el in completions]
#         results = []
#         for i, sample in enumerate(audio):
#             for remove_string in ["<|endoftext|>"]:
#                 if remove_string in text_completions[i]:
#                     print(f"REMOVING: {remove_string} from: {text_completions[i]}")
#                     text_completions[i] = text_completions[i].replace(remove_string, "")

#             text = get_text_normal(asr_model, sample[0], text_completions[i])
#             print(text_completions[i])
#             print(text)
#             results.append(text)
#             print('--')
#         # print([el.shape for el in audio])
#         print(answer)
        
#         wers = [word_error_rate_detail([r], [a], use_cer=False, normalize = True)[0] for r, a in zip(results, answer)]
#         improvements = [original_wer[i] - wers[i] for i in range(len(wers))]
#         print(improvements)
#         #print(completions[0][0])
#         return improvements
#     return reward_num_unique_chars

@torch.no_grad()
def reward_func(asr_model, processor, device):
    def reward_num_unique_chars(prompts, answer, completions, audio, original_wer, **kwargs):
        text_completions = [el[0]['content'] for el in completions]
        # print([el.shape for el in audio])
        # exit()
        input_features = processor.feature_extractor(
            [el.squeeze(0) for el in audio],
            return_tensors="pt",
            sampling_rate=16_000,
        )
        decoder_ids = []
        label_ids = []
        for completion, answer in zip(text_completions, answer):
            prompt_ids = processor.tokenizer.get_prompt_ids(completion).tolist()
            answer_ids = processor.tokenizer(answer).input_ids
            combined_ids = prompt_ids + answer_ids
            decoder_ids.append(combined_ids)
            label_ids.append([processor.tokenizer.pad_token_id] * len(prompt_ids) + answer_ids)
    
        decoder_padded = processor.tokenizer.pad(
            {"input_ids": decoder_ids},
            padding=True,
            return_tensors="pt",
        )
        decoder_label_padded = processor.tokenizer.pad(
            {"input_ids": label_ids},
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        # print(label_ids[0])
        # print(decoder_label_padded[0])
        # exit()
        decoder_label_padded[decoder_label_padded == processor.tokenizer.pad_token_id] = -100
        decoder_label_padded = decoder_label_padded.to(device)
        mask = decoder_label_padded != -100

        encoder_outputs = asr_model.model.encoder(input_features=input_features['input_features'].to(device, dtype=torch.bfloat16))

        decoder_input_ids = modeling_whisper.shift_tokens_right(
            decoder_padded['input_ids'].to(device), asr_model.config.pad_token_id, asr_model.config.decoder_start_token_id
        )

        outputs = asr_model(
            encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids, 
            #decoder_attention_mask=decoder_padded['attention_mask'].to(device),
            labels=decoder_label_padded
        )
        loss = outputs.loss.sum(-1) / mask.sum(-1)
        reward = -loss

        return reward.tolist()
    return reward_num_unique_chars


def get_dataset(args, tokenizer, max_files=-1): # preload all text
    dataset = []
    for path in args.dataset_paths:
        if not os.path.exists(path):
            raise ValueError(f"Dataset path {path} does not exist.")
        files = os.listdir(path)
        for i, file_path in enumerate(tqdm(files)):
            if max_files > 0 and i > max_files: break
            full_file_path = os.path.join(path, file_path)
            dataset.append(full_file_path)
    dataset = CustomDataset(tokenizer, dataset)
    return dataset


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    accelerator = Accelerator()
    
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = get_dataset(args, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(args.language_model, torch_dtype=torch.bfloat16).to("cuda")

    processor, asr_model = get_whisper(args.whisper_model, device=args.whisper_device, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    #asr_model = whisper.load_model(args.whisper_model, device=args.whisper_device)
    #asr_model = WhisperModel(args.whisper_model, device=args.whisper_device, compute_type="float16", device_index=1)


    if not args.no_wandb and accelerator.is_local_main_process: wandb.init(project="synctxasr")


    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_generations=16,
        warmup_steps=100,
        max_prompt_length=1024,
        max_completion_length=98,
        temperature=0.9,
        gradient_checkpointing=False,
        num_train_epochs=args.epochs,
        #eval_strategy="steps",
        save_steps=1000,
        #eval_steps=1000,
        max_grad_norm=0.1,
        report_to=["wandb"] if not args.no_wandb else None,
        log_on_each_node=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=4,
        optim="adafactor",
        use_vllm=False,
        log_completions=True,
        beta=4e-2,
    
    )

  


    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_func(asr_model, processor, args.whisper_device),
        ],
        args=training_args,
        train_dataset=train_dataset,
        #eval_dataset=dev_dataset,
        #peft_config=peft_config
    )

    #trainer.evaluate()
    try: trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        print("Saving model before exiting...")
        trainer.save_state()
        trainer.save_model(os.path.join(args.output_dir, "interrupted_model"))
        raise e
        
    trainer.save_model(os.path.join(args.output_dir, "final_model"))

    if not args.no_wandb and accelerator.is_local_main_process:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', type=str, default='/store/store4/data/huggingface_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--dataset_paths', nargs='+', type=str, default=[
        '/store/store4/data/TEDLIUM3_Whisper_tiny_en_outputs/train',
        '/store/store4/data/LIBRISPEECH_Whisper_tiny_en_outputs/train_other/train-other-500/',
        # '/store/store4/data/LIBRISPEECH_Whisper_tiny_en_outputs/train_clean_100/train-clean-100/'
    ])
    
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--batch_size', type=int, default=16) # 32
    parser.add_argument('--lr', type=float, default=2e-7)
    parser.add_argument('--output_dir', type=str, default='/store/store5/data/acp21rjf_checkpoints/synctxasr/grpo/b4')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2) # 1
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--whisper_model', type=str, default='openai/whisper-tiny.en')
    parser.add_argument('--whisper-device', type=str, default='cuda:1') # cuda:1

    args = parser.parse_args()    
    if not torch.cuda.is_available(): args.device = 'cpu'

    main(args)
