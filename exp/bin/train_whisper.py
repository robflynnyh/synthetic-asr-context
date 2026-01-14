import argparse
from tqdm import tqdm
from synctxasr.data import dataset_functions
from synctxasr.wer import word_error_rate_detail
from synctxasr.misc import int_or_none

from accelerate import Accelerator

import torch
import os
from functools import partial
import wandb
from synctxasr.wer import word_error_rate_detail    
import random
#from faster_whisper import WhisperModel
from madgrad import MADGRAD
from itertools import chain, repeat
from einops import repeat as einops_repeat
from einops import rearrange
from torch import nn
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
import gc
from transformers import WhisperForCausalLM
import re

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # doesn't matter, vLLM doesn't check
)

def remove_first_thinking_tags(s):
    # Remove only the first occurrence of (thinking) tags and their content, but keep the brackets
    return re.sub(r'\(thinking\).*?\(.thinking\)', '', s, count=1, flags=re.DOTALL)

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
            loss_fct = CrossEntropyLoss(reduce=False, ignore_index=-100, reduction='none')
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
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    # @staticmethod
    # def format_example(example, mode='thinking'):
    #     if mode == 'thinking':
    #         prompt_messages = [
    #             {"role": "system", "content": "You are a helpful assistant. The user will supply a sentence, please output a short summary of the sentence."},
    #             {"role": "user", "content": f"'{example}'."},
    #         ]
    #     elif mode == 'previous':
    #         prompt_messages = [
    #             {"role": "system", "content": "You are a helpful assistant. The user will supply a sentence, please output a plausible previous sentence."},
    #             {"role": "user", "content": f"'{example}'."},
    #         ]
    #     else:
    #         raise ValueError("Invalid mode. Choose 'thinking' or 'previous'.")
        
    #     return prompt_messages
    
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
        
        audio = data['waveform'].squeeze(0).numpy()
        gold_text = data['text']
        generation = data['generation']
        wer = word_error_rate_detail([generation], [gold_text], use_cer=False, normalize=True)[0] * 100

        return {
            'answer': gold_text,
            'audio': audio,
            'wer': wer,
        }
    
def collator(batch, processor):
    audio = [el['audio'] for el in batch]
    gold_text = [el['answer'] for el in batch]
    wers = [el['wer'] for el in batch]
    wer = torch.tensor(wers)
    #print([el.shape for el in audio])
    input_features = processor.feature_extractor(
        audio,
        return_tensors="pt",
        sampling_rate=16_000,
    )

    return {
        'input_features': input_features['input_features'],
        'answer': gold_text,
        'wer': wer,
    }

def get_templated_transcriptions(transcriptions):
    templated_transcriptions = []
    untemplated_transcriptions = []
    for transcription in transcriptions:
        cur_transcription = transcription.strip()
        if cur_transcription.startswith('(thinking)') and '(.thinking)' in cur_transcription:
            cur_transcription = remove_first_thinking_tags(cur_transcription)
            templated_transcriptions.append(transcription.strip())
            untemplated_transcriptions.append(cur_transcription.strip())
        else:
            untemplated_transcriptions.append(transcription.strip())
            templated_transcriptions.append(f"(thinking)(.thinking) {transcription.strip()}")
    return templated_transcriptions, untemplated_transcriptions
            

def template(transcription):
    response = client.chat.completions.create(
        model="/store/store4/data/huggingface_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. The user will supply you with a sentence that was generated by an ASR model. Your task is to predict up to 5 related comma seperated keywords. Do not include any other information in your response."},
            {"role": "user", "content": f"'{transcription}'."},
        ],
        max_tokens=30,
    )
    response = response.choices[0].message.content
    return f"(thinking) {response} (.thinking) {transcription.strip()}"

def remove_templates(transcriptions):
    untemplated_transcriptions = []
    for transcription in transcriptions:
        cur_transcription = transcription.strip()
        if cur_transcription.startswith('(thinking)') and '(.thinking)' in cur_transcription:
            cur_transcription = remove_first_thinking_tags(cur_transcription)
            untemplated_transcriptions.append(cur_transcription.strip())
        elif cur_transcription.startswith('(thinking)'):
            cur_transcription = "" # did not find the end tag
            untemplated_transcriptions.append(cur_transcription.strip())
        else:
            untemplated_transcriptions.append(transcription.strip())
    return untemplated_transcriptions

def get_tag_reward(transcriptions):
    rewards = []
    for transcription in transcriptions:
        if transcription.startswith('(thinking)') and '(.thinking)' in transcription:
            rewards.append(1)
        else:
            rewards.append(0)
    return torch.tensor(rewards)
    
            

def train(args, model, processor, optim, dataloader, original_model):
    repeats = 12
    temp = (0.3, 0.4, 0.6, 0.8, 1.0)
    warmup_steps = 2500
    completed_warmup = 2500
    save_every = 1000


    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(dataloader)):
            try:
                audio = batch['input_features'].to(args.device)
                gold_text = batch['answer']
                old_wers = batch['wer']

                if completed_warmup < warmup_steps:
                    
                    with torch.no_grad():
                        encoder_outputs = original_model.model.encoder(input_features=audio)
                        predicted_ids = original_model.generate(
                            input_features = audio,
                            attention_mask = None,
                            prompt_ids = None,
                            return_timestamps=False,
                            is_multilingual=False,
                            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                            do_sample=True,
                            compression_ratio_threshold=2.4,
                            num_beams=5,
                            logprob_threshold=-1,
                            max_length=400,
                            encoder_outputs=encoder_outputs,
                        )
                        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                        templated_transcriptions = [template(el) for el in transcription]
                        print(templated_transcriptions)
                       
                        labels = processor.tokenizer(
                            templated_transcriptions,
                            return_tensors="pt",
                            padding=True,
                            truncation=False,
                        ).input_ids.to(args.device)
                        labels[labels == processor.tokenizer.pad_token_id] = -100
                        mask = labels != -100

                    encoder_outputs = model.model.encoder(input_features=audio)
                    outputs = model(encoder_outputs=encoder_outputs, labels=labels)
                    optim.zero_grad()
                    loss = outputs.loss 
                    
                    loss = loss.sum() / mask.sum()
                    print(loss.item())

                    loss.backward()
                    model.model.decoder.embed_tokens.weight.grad[:-2] = 0.0 # we're only training the new tokens!
                    model.proj_out.weight.grad[:-2] = 0.0 # we're only training the new tokens!
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()

                    completed_warmup += 1
                    if completed_warmup == warmup_steps:
                        del original_model # no longer needed
                        try:
                            model.save_pretrained("./whisper-tiny-finetuned_0")
                            processor.save_pretrained("./whisper-tiny-finetuned_0")
                        except Exception as e:
                            print(f"Error saving model: {e}")
                            continue        

                else:                    
                    with torch.no_grad():
                        encoder_outputs = model.model.encoder(input_features=audio)
                        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.repeat_interleave(repeats, dim=0).clone()

                        predicted_ids = model.generate(
                            input_features = audio.repeat_interleave(repeats, dim=0).clone(),
                            attention_mask = None,
                            prompt_ids = None,
                            return_timestamps=False,
                            is_multilingual=False,
                            temperature=temp,
                            do_sample=True,
                            compression_ratio_threshold=2.4,
                            num_beams=1,
                            logprob_threshold=-1,
                            max_new_tokens=200,
                            no_repeat_ngram_size=5, 
                            encoder_outputs=encoder_outputs,   
                        )
                        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

                        #has_tags = get_tag_reward(transcription)
                        untemplated_transcriptions = remove_templates(transcription)
                        print(gold_text)
                        print('--')
                        print(untemplated_transcriptions)
                        print('--')
    
                        repeated_gold_text = list(chain.from_iterable(repeat(x, repeats) for x in gold_text))

                        print(repeated_gold_text)

                        wers = [
                            word_error_rate_detail([hyp], [ref], use_cer=False, normalize=True)[0] * 100
                            for hyp, ref in zip(untemplated_transcriptions, repeated_gold_text)
                        ]   
                        wers = torch.tensor(wers)
                        print(transcription)
                        
                        # rearange into groups of repeats
                        wers = rearrange(wers, '(b r) -> b r', b=len(wers)//repeats, r=repeats)
                       # has_tags = rearrange(has_tags, '(b r) -> b r', b=len(has_tags)//repeats, r=repeats)
                        print(wers)
                        print(old_wers)
                        reward_mask = ((old_wers.unsqueeze(-1) >= wers).sum(-1) > 0).float().unsqueeze(-1)
                        
                        wers = wers # - has_tags * 10

                        #wers[wers != 0] = 1
                        
                        std_wer = wers.std(dim=1, keepdim=True)
                        mean_wer = wers.mean(dim=1, keepdim=True)
                        normed_wers = ((wers - mean_wer) / std_wer)
                        # replace all nan values with 0
                        normed_wers[torch.isnan(normed_wers)] = 0
                        normed_wers[torch.isinf(normed_wers)] = 0
                        reward = normed_wers * -1 # because we want to minimize the WER
                        reward = reward * reward_mask # only apply the reward to the samples that are better or equal to the original WER (prevents degradation)
                        print(reward)
                        #print(f"Transcription: {wers}")
    
               
                        
        
                        
                        labels = processor.tokenizer(
                            transcription,
                            return_tensors="pt",
                            padding=True,
                            truncation=False,
                        ).input_ids.to(args.device)

                        

                        labels[labels == processor.tokenizer.pad_token_id] = -100
                        mask = labels != -100

                    outputs = model(encoder_outputs=encoder_outputs, labels=labels)

                    optim.zero_grad()
                    #print(outputs.loss[0])
                    # print(labels.shape)
                    # print(outputs.loss.shape)
                    # exit()
                    loss = outputs.loss * reward.flatten().unsqueeze(-1).to(outputs.loss.device)
                    
                    loss = loss.sum() / mask.sum()
                    print(loss.item())

                    loss.backward()
                    model.model.decoder.embed_tokens.weight.grad[:-2] = 0.0 # we're only training the new tokens!
                    model.proj_out.weight.grad[:-2] = 0.0 # we're only training the new tokens!
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    
                    optim.step()

                    if i % save_every == 0 and i > 0:
                        try:
                            # remove dir if exists
                            model.save_pretrained(f"./whisper-tiny-finetuned_{i}")
                            processor.save_pretrained(f"./whisper-tiny-finetuned_{i}")
                        except Exception as e:
                            print(f"Error saving model: {e}")
                            continue

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("Out of memory error, skipping batch")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                elif "invalid for input" in str(e):
                    print("Invalid input error, skipping batch")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
                
    print("Training complete")
    # Save the model
    model.save_pretrained("./whisper-tiny-finetuned")
    processor.save_pretrained("./whisper-tiny-finetuned")
    # Save the processor
    # Save the tokenizer




           
            #print(outputs.logits.shape)


def get_dataset(args, max_files=-1): # preload all text
    dataset = []
    for path in args.dataset_paths:
        if not os.path.exists(path):
            raise ValueError(f"Dataset path {path} does not exist.")
        files = os.listdir(path)
        for i, file_path in enumerate(tqdm(files)):
            if max_files > 0 and i > max_files: break
            full_file_path = os.path.join(path, file_path)
            dataset.append(full_file_path)
    dataset = CustomDataset(dataset)
    return dataset

def get_training_modules(model):
    # we are only training the feedforward fc1 layers
    for param in model.parameters():
        param.requires_grad = False

    params = []
    decoder_layers = model.model.decoder.layers
    for layer in decoder_layers:
        # for param in layer.fc1.parameters():
        #     param.requires_grad = True
        # params.extend(layer.fc1.parameters())

        # for param in layer.encoder_attn.parameters():
        #     param.requires_grad = True
        # params.extend(layer.encoder_attn.parameters())
        for param in layer.self_attn.parameters():
            param.requires_grad = True
        params.extend(layer.self_attn.parameters())

    for param in model.proj_out.parameters():
        param.requires_grad = True
    for param in model.model.decoder.embed_tokens.parameters():
        param.requires_grad = True
    params.extend(model.model.decoder.embed_tokens.parameters())
  
    return params


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_dataset = get_dataset(args)

    processor, model = get_whisper(args.whisper_model, device=args.device)

    new_tokens = ["(thinking)", "(.thinking)"]
    processor.tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))

    # checkpoint = 'whisper-tiny-finetuned_0'
    # # load the model from the checkpoint
    # loaded_model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    # model.load_state_dict(loaded_model.state_dict(), strict=False)

    _, original_model = get_whisper("openai/whisper-small.en", device=args.device)



    
    optim = MADGRAD(
        get_training_modules(model),
        lr=args.lr,
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=partial(collator, processor=processor),
        num_workers=8,
        pin_memory=False,
    )

    train(args, model, processor, optim, dataloader, original_model)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', type=str, default='/store/store4/data/huggingface_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--dataset_paths', nargs='+', type=str, default=[
        '/store/store4/data/TEDLIUM3_Whisper_tiny_en_outputs/train',
        '/store/store4/data/LIBRISPEECH_Whisper_tiny_en_outputs/train_other/train-other-500/',
        '/store/store4/data/LIBRISPEECH_Whisper_tiny_en_outputs/train_clean_100/train-clean-100/'
    ])
    
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--batch_size', type=int, default=4) # 32
    parser.add_argument('--lr', type=float, default=1e-8)
    parser.add_argument('--output_dir', type=str, default='/store/store5/data/acp21rjf_checkpoints/synctxasr/grpo/b3')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2) # 1
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--whisper_model', type=str, default='openai/whisper-tiny.en')
    parser.add_argument('--whisper-device', type=str, default='cuda:1') # cuda:1

    args = parser.parse_args()    
    if not torch.cuda.is_available(): args.device = 'cpu'

    main(args)
