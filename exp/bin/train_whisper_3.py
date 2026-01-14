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
            #print('--',lm_logits.softmax(-1)[0,:, 50257])
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
        history = data['previous_generation']
        return {
            'answer': gold_text,
            'audio': audio,
            'wer': wer,
            'history': history,
        }
    
def collator(batch, processor):
    audio = [el['audio'] for el in batch]
    gold_text = [el['answer'] for el in batch]
    wers = [el['wer'] for el in batch]
    history = [el['history'] for el in batch]
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
        'history': history,
    }




def train(args, model, processor, optim, dataloader):
    repeats = 12
    temp = (0.4, 0.6, 0.8, 1.0)
    save_every = 1000


    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(dataloader)):
            try:
                audio = batch['input_features'].to(args.device)
                gold_text = batch['answer']
                old_wers = batch['wer']
                history = batch['history']

                is_corrupted = False
                if random.random() < 0.05:
                    if random.random() < 0.5:
                        audio[0] = torch.randn_like(audio[0]) * audio[0].std() + audio[0].mean() 
                    elif random.random() < 0.5:
                        audio[0] = torch.zeros_like(audio[0])
                    else:
                        audio[0] = torch.randn_like(audio[0]) * 0.1

                    gold_text[0] = ""
                    old_wers[0] = 0.0
                    is_corrupted = True
                    print(f'Corrupted audio zero')

                with torch.no_grad():
                    encoder_outputs = model.model.encoder(input_features=audio)
                    encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.repeat_interleave(repeats, dim=0).clone()
                    repeated_history = list(chain.from_iterable(repeat(x, repeats) for x in history)) 
                    
                    prompt_id_lists = [processor.get_prompt_ids(el, return_tensors=None) if el != None else [] for el in repeated_history]
                    prompt_ids = [torch.tensor(el) for el in prompt_id_lists]
                    prompt_ids = [el.to(args.device) for el in prompt_ids]
                

                    predicted_ids = model.generate(
                        input_features = audio.repeat_interleave(repeats, dim=0).clone(),
                        attention_mask = None,
                        prompt_ids = prompt_ids, # only works like this due to monkey patching in synctxasr.asr!
                        return_timestamps=False,
                        is_multilingual=False,
                        temperature=temp,
                        do_sample=True,
                        compression_ratio_threshold=2.4,
                        num_beams=1,
                        logprob_threshold=-1,
                        max_new_tokens=200,
                        no_repeat_ngram_size=5, 
                        prompt_condition_type='first-segment',
                        condition_on_prev_tokens=True,  
                    )
                    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                    if is_corrupted:
                        transcription[0] = ""

                    #has_tags = get_tag_reward(transcription)
                    print(gold_text)
                    print('--')
                    print(history)
                    print('--')
                    print(transcription)
                    print('--')

                    repeated_gold_text = list(chain.from_iterable(repeat(x, repeats) for x in gold_text))

                    print(repeated_gold_text)

                    wers = [
                        word_error_rate_detail([hyp], [ref], use_cer=False, normalize=True)[0] * 100
                        for hyp, ref in zip(transcription, repeated_gold_text)
                    ]   
                    wers = torch.tensor(wers)
                    print(transcription)
                    
                    # rearange into groups of repeats
                    wers = rearrange(wers, '(b r) -> b r', b=len(wers)//repeats, r=repeats)

                    # has_tags = rearrange(has_tags, '(b r) -> b r', b=len(has_tags)//repeats, r=repeats)
                    print(wers)
                    print(old_wers)
                    reward_mask_a = ((old_wers.unsqueeze(-1) > wers).sum(-1) > 0).float().unsqueeze(-1)
                    reward_mask_b = ((wers == 0.0).sum(-1) > 0).float().unsqueeze(-1)
                 
                    combined = (reward_mask_a + reward_mask_b).clamp(0, 1).bool().float() # if either is true, we want to apply the reward
                    reward_mask = combined

                    #wers[wers != 0] = 1

                    min_wer = wers.min(dim=1, keepdim=True)
                    max_wer = wers.max(dim=1, keepdim=True)
                    num_min = (wers == min_wer.values).sum(dim=1, keepdim=True)
                    new_reward = wers.clone()
                    #new_reward[wers != min_wer.values] = 0

                    ''' A positive reward of 1.0 will be assigned to the lowest wer, this reward is divided equally for all generations that get that wer
                        A negatice reward of -1.0 will be assigned to all other generations, this is divided based on the wer difference 
                    '''
                    for b in range(new_reward.shape[0]):
                        b_min = num_min[b].item()
                        if min_wer.values[b] == max_wer.values[b] and min_wer.values[b] != 0:
                            new_reward[b] *= 0
                        elif min_wer.values[b] == max_wer.values[b] and min_wer.values[b] == 0:
                            new_reward[b] = 1/b_min
                        else:
                            new_reward[b][wers[b] == min_wer.values[b]] = 1 / b_min
                            diff = min_wer.values[b] - wers[b][wers[b] != min_wer.values[b]]
                            diff = diff / abs(sum(diff))
                            new_reward[b][wers[b] != min_wer.values[b]] = diff


                    reward = new_reward * reward_mask

                    # std_wer = wers.std(dim=1, keepdim=True)
                    # mean_wer = wers.mean(dim=1, keepdim=True)
                    # normed_wers = ((wers - mean_wer) / std_wer)
                    # # replace all nan values with 0
                    # normed_wers[torch.isnan(normed_wers)] = 0
                    # normed_wers[torch.isinf(normed_wers)] = 0
                    # reward = normed_wers * -1 # because we want to minimize the WER
                    # reward = reward * reward_mask # only apply the reward to the samples that are better or equal to the original WER (prevents degradation)
                    print(reward)
                    #print(f"Transcription: {wers}")

                    decoder_ids, label_ids = [], []
                    prompt_masks = []
               
                    for prompt, cur_trans in zip(prompt_id_lists, transcription):
                        transcript_ids = processor.tokenizer(cur_trans).input_ids
                        combined_ids = prompt + transcript_ids
                      
                        decoder_ids.append(combined_ids)
                        label_ids.append(combined_ids)
                        prompt_masks.append([True] * len(prompt) + [False] * len(transcript_ids))

                    # pad prompt masks
                    max_length = max(map(len, decoder_ids))
                    prompt_masks = [mask + [False] * (max_length - len(mask)) for mask in prompt_masks]
                    prompt_masks = torch.tensor(prompt_masks).to(args.device)
        

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

                    shifted_labels = torch.zeros_like(decoder_label_padded)
                    shifted_labels[:, :-1] = decoder_label_padded[:, 1:].clone()
                    shifted_labels[:, -1] = processor.tokenizer.pad_token_id
                    
                    shifted_labels[shifted_labels == processor.tokenizer.pad_token_id] = -100
                    shifted_labels = shifted_labels.to(args.device)
                    mask = shifted_labels != -100
                    
                    decoder_input_ids = decoder_padded['input_ids'].to(args.device)

                outputs = model(
                    encoder_outputs=encoder_outputs, 
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_padded['attention_mask'].to(args.device),
                    labels=shifted_labels,
                )
                #print(outputs.loss[0])

                optim.zero_grad()
                #print(outputs.loss[0])
                # print(labels.shape)
                # print(outputs.loss.shape)
                # exit()
                #print(outputs.loss)
              

                reward = reward.flatten().unsqueeze(-1).to(outputs.loss.device)
                reward = reward * (~prompt_masks).float() + prompt_masks.float() * (1/repeats)
                reward = reward * reward_mask.repeat_interleave(repeats, dim=0).to(reward.device)
                loss = outputs.loss * reward

                loss = loss.sum() / mask.sum()
                print(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)                
                optim.step()

                if random.random() < 0.2:
                    start_strings = ["The", "A", "An", "So", "Basically", "Well", ""]
                    sstring = random.choice(start_strings)
                    decoder_inputs = torch.tensor(processor.get_prompt_ids(sstring, return_tensors=None)).to(args.device).unsqueeze(0)
                    encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[0, None]
                    predicted_ids = model.generate(
                        input_features = audio[0, None],
                        attention_mask = None,
                        return_timestamps=False,
                        is_multilingual=False,
                        temperature=(0.3, 0.4, 0.6),
                        do_sample=True,
                        compression_ratio_threshold=2.4,
                        num_beams=1,
                        logprob_threshold=-1,
                        max_new_tokens=200,
                        no_repeat_ngram_size=5, 
                        encoder_outputs=encoder_outputs, 
                        decoder_input_ids=decoder_inputs,
                        return_dict_in_generate=True,
                        disable_logits_processor=True
                    )
                    transcription = processor.tokenizer.batch_decode(predicted_ids['sequences'], skip_special_tokens=False)[0]
                    print(f'with: {sstring}: {transcription}')

                if i % save_every == 0 and i > 0:
                    try:
                        # remove dir if exists
                        save_path = os.path.join(args.output_dir, f"whisper-tiny-finetuned_{i}")
                        model.save_pretrained(save_path)
                        processor.save_pretrained(save_path)
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
    save_path = os.path.join(args.output_dir, f"whisper-tiny-finetuned_final")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
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
        for param in layer.self_attn.parameters():
            param.requires_grad = True
        params.extend(layer.self_attn.parameters())
        for param in layer.encoder_attn.parameters():
            param.requires_grad = True
        params.extend(layer.encoder_attn.parameters())
    # for param in model.model.decoder.embed_tokens.parameters():
    #     param.requires_grad = True
    # params.extend(model.model.decoder.embed_tokens.parameters())
    # for param in model.proj_out.parameters():
    #     param.requires_grad = True
    # params.extend(model.proj_out.parameters())
  
    return params


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_dataset = get_dataset(args)

    processor, model = get_whisper(args.whisper_model, device=args.device)


    
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

    train(args, model, processor, optim, dataloader)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--dataset_paths', nargs='+', type=str, default=[
        '/store/store4/data/TEDLIUM3_Whisper_tiny_en_outputs/train',
        '/store/store4/data/LIBRISPEECH_Whisper_tiny_en_outputs/train_other/train-other-500/',
    ])
    
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--batch_size', type=int, default=4) # 32
    parser.add_argument('--lr', type=float, default=1e-8)
    parser.add_argument('--output_dir', type=str, default='/store/store5/data/acp21rjf_checkpoints/synctxasr/whisper/1/')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2) # 1
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--whisper_model', type=str, default='openai/whisper-tiny.en')
    parser.add_argument('--whisper-device', type=str, default='cuda:1') # cuda:1

    args = parser.parse_args()    
    if not torch.cuda.is_available(): args.device = 'cpu'

    main(args)
