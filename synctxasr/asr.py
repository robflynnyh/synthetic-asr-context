import torch, numpy as np 
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper import generation_whisper
from typing import List, Dict, Any
from torch.nn import functional as F
from typing import Optional, Callable, Union, Tuple
import warnings

def generate(
    self,
    input_features: Optional[torch.Tensor] = None,
    generation_config: Optional[generation_whisper.GenerationConfig] = None,
    logits_processor: Optional[generation_whisper.LogitsProcessorList] = None,
    stopping_criteria: Optional[generation_whisper.StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: bool = False,
    return_timestamps: Optional[bool] = None,
    task: Optional[str] = None,
    language: Optional[Union[str, List[str]]] = None,
    is_multilingual: Optional[bool] = None,
    prompt_ids: Optional[torch.Tensor] = None,
    prompt_condition_type: Optional[str] = None,  # first-segment, all-segments
    condition_on_prev_tokens: Optional[bool] = None,
    temperature: Optional[Union[float, Tuple[float, ...]]] = None,
    compression_ratio_threshold: Optional[float] = None,
    logprob_threshold: Optional[float] = None,
    no_speech_threshold: Optional[float] = None,
    num_segment_frames: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    time_precision: float = 0.02,
    time_precision_features: float = 0.01,
    return_token_timestamps: Optional[bool] = None,
    return_segments: bool = False,
    return_dict_in_generate: Optional[bool] = None,
    force_unique_generate_call: Optional[bool] = None,
    disable_logits_processor=False,
    **kwargs,
):
    
    # 0. deprecate old inputs
    if "inputs" in kwargs:
        input_features = kwargs.pop("inputs")
        warnings.warn(
            "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
            FutureWarning,
        )

    # 1. prepare generation config
    generation_config, kwargs = self._prepare_generation_config(generation_config, **kwargs)

    # 2. set global generate variables
    input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
    num_segment_frames = input_stride * self.config.max_source_positions
    batch_size, total_input_frames = self._retrieve_total_input_frames(
        input_features=input_features, input_stride=input_stride, kwargs=kwargs
    )
    is_shortform = total_input_frames <= num_segment_frames

    # 3. Make sure generation config is correctly set
    # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
    return_dict_in_generate = self._set_return_outputs(
        return_dict_in_generate=return_dict_in_generate,
        return_token_timestamps=return_token_timestamps,
        logprob_threshold=logprob_threshold,
        generation_config=generation_config,
    )
    timestamp_begin = self._set_return_timestamps(
        return_timestamps=return_timestamps, is_shortform=is_shortform, generation_config=generation_config
    )
    self._set_language_and_task(
        language=language, task=task, is_multilingual=is_multilingual, generation_config=generation_config
    )
    self._set_num_frames(
        return_token_timestamps=return_token_timestamps, generation_config=generation_config, kwargs=kwargs
    )
    self._set_thresholds_and_condition(
        generation_config=generation_config,
        logprob_threshold=logprob_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_prev_tokens=condition_on_prev_tokens,
    )
    self._set_prompt_condition_type(
        generation_config=generation_config,
        prompt_condition_type=prompt_condition_type,
    )

    # pass self.config for backward compatibility
    init_tokens = self._retrieve_init_tokens(
        input_features,
        batch_size=batch_size,
        generation_config=generation_config,
        config=self.config,
        num_segment_frames=num_segment_frames,
        kwargs=kwargs,
    )
    # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
    # where the input ids are handled explicitly by the generate method
    self._check_decoder_input_ids(kwargs=kwargs)

    # 3. Retrieve logits processors
    device = kwargs["encoder_outputs"][0].device if "encoder_outputs" in kwargs else input_features.device
    begin_index = init_tokens.shape[1]
    num_beams = kwargs.get(
        "num_beams",
        generation_config.num_beams
        if hasattr(generation_config, "num_beams") and generation_config.num_beams is not None
        else 1,
    )
    if "assistant_model" in kwargs:
        # speculative decoding: the model should be able to return eos token
        generation_config.begin_suppress_tokens = None

    logits_processor = self._retrieve_logit_processors(
        generation_config=generation_config,
        logits_processor=logits_processor,
        begin_index=begin_index,  # begin index is index of first generated decoder token
        num_beams=num_beams,
        device=device,
    )

    # 4 Set and retrieve global generation variables
    self._set_condition_on_prev_tokens(
        condition_on_prev_tokens=condition_on_prev_tokens, generation_config=generation_config
    )

    temperatures = [temperature] if not isinstance(temperature, (list, tuple)) else temperature
    temperature = temperatures[0]

    max_frames, seek = self._retrieve_max_frames_and_seek(
        batch_size=batch_size,
        attention_mask=attention_mask,
        total_input_frames=total_input_frames,
        is_shortform=is_shortform,
    )

    # 5 Prepare running variables, list for generation
    num_return_sequences = generation_config.num_return_sequences
    (
        batch_idx_map,
        cur_bsz,
        input_features,
        seek,
        max_frames,
        init_tokens,
        do_condition_on_prev_tokens,
    ) = self._expand_variables_for_generation(
        input_features=input_features,
        seek=seek,
        max_frames=max_frames,
        init_tokens=init_tokens,
        batch_size=batch_size,
        condition_on_prev_tokens=condition_on_prev_tokens,
        generation_config=generation_config,
    )

    current_segments = self._prepare_segments(
        prompt_ids=prompt_ids,
        batch_size=cur_bsz,
        generation_config=generation_config,
    )
    # 5bis speculative decoding: ensure the assistant model does only one call to generate and therefore returns decoder input token ids and eos token id
    # we set a flag in the generation config to force the model to make only one call to generate and return the decoder input token ids and eos token id
    if "assistant_model" in kwargs:
        assistant_model = kwargs["assistant_model"]
        assistant_model.generation_config.force_unique_generate_call = True

    if force_unique_generate_call is None:
        if hasattr(generation_config, "force_unique_generate_call"):
            force_unique_generate_call = generation_config.force_unique_generate_call
        elif hasattr(self.generation_config, "force_unique_generate_call"):
            force_unique_generate_call = self.generation_config.force_unique_generate_call
        else:
            force_unique_generate_call = False

    # 6 Transcribe audio until we reach the end of all input audios
    while (seek < max_frames).any():
        # 6.1 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
        # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
        # to know which original audio is being decoded
        # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
        input_features, cur_bsz, batch_idx_map = self._maybe_reduce_batch(
            input_features=input_features,
            seek=seek,
            max_frames=max_frames,
            cur_bsz=cur_bsz,
            batch_idx_map=batch_idx_map,
        )
        time_offset = (
            seek.to(torch.float32 if device.type == "mps" else torch.float64) * time_precision / input_stride
        )
        seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

        # 6.2 cut out next 30s segment from input features
        segment_input = self._get_input_segment(
            input_features=input_features,
            seek=seek,
            seek_num_frames=seek_num_frames,
            num_segment_frames=num_segment_frames,
            cur_bsz=cur_bsz,
            batch_idx_map=batch_idx_map,
        )

        # 6.3 prepare decoder input ids
        suppress_tokens = generation_whisper._get_attr_from_logit_processors(
            logits_processor, generation_whisper.SuppressTokensLogitsProcessor, "suppress_tokens"
        )

        decoder_input_ids, kwargs = self._prepare_decoder_input_ids(
            cur_bsz=cur_bsz,
            init_tokens=init_tokens,
            current_segments=current_segments,
            batch_idx_map=batch_idx_map,
            do_condition_on_prev_tokens=do_condition_on_prev_tokens,
            prompt_ids=prompt_ids,
            generation_config=generation_config,
            config=self.config,
            device=init_tokens.device,
            suppress_tokens=suppress_tokens,
            timestamp_begin=timestamp_begin,
            kwargs=kwargs,
        )

        # 6.4 set max new tokens or max length
        self._set_max_new_tokens_and_length(
            config=self.config,
            decoder_input_ids=decoder_input_ids,
            generation_config=generation_config,
        )

        # 6.5 Set current `begin_index` for all logit processors
        if logits_processor is not None:
            for proc in logits_processor:
                if hasattr(proc, "set_begin_index"):
                    proc.set_begin_index(decoder_input_ids.shape[-1])

    
        if disable_logits_processor: logits_processor = None # enables generatio of start of transcript tokens
        # 6.6 Run generate with fallback
        (
            seek_sequences,
            seek_outputs,
            should_skip,
            do_condition_on_prev_tokens,
            model_output_type,
        ) = self.generate_with_fallback(
            segment_input=segment_input,
            decoder_input_ids=decoder_input_ids,
            cur_bsz=cur_bsz,
            batch_idx_map=batch_idx_map,
            seek=seek,
            num_segment_frames=num_segment_frames,
            max_frames=max_frames,
            temperatures=temperatures,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            return_token_timestamps=return_token_timestamps,
            do_condition_on_prev_tokens=do_condition_on_prev_tokens,
            is_shortform=is_shortform,
            batch_size=batch_size,
            attention_mask=attention_mask,
            kwargs=kwargs,
        )

        # 6.7 In every generated sequence, split by timestamp tokens and extract segments
        for i, seek_sequence in enumerate(seek_sequences):
            prev_i = batch_idx_map[i]

            if should_skip[i]:
                seek[prev_i] += seek_num_frames[prev_i]
                continue

            segments, segment_offset = self._retrieve_segment(
                seek_sequence=seek_sequence,
                seek_outputs=seek_outputs,
                time_offset=time_offset,
                timestamp_begin=timestamp_begin,
                seek_num_frames=seek_num_frames,
                time_precision=time_precision,
                time_precision_features=time_precision_features,
                input_stride=input_stride,
                prev_idx=prev_i,
                idx=i,
                return_token_timestamps=return_token_timestamps,
                decoder_input_ids=decoder_input_ids,
            )

            seek[prev_i] += segment_offset

            current_segments[prev_i] += segments

        if force_unique_generate_call:
            break

    # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
    # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
    # if verbose:
    #     print(current_segments[0][0]['tokens'])
    final_segments = (
        [x[1:] for x in current_segments]
        if (prompt_ids is not None and generation_config.prompt_condition_type == "first-segment")
        else current_segments
    )

    # if return_dict_in_generate=True and we forced a unique call to generate or return_timestamps=False, meaning we are sure only one call to generate has been made,
    # -> we can return a ModelOutput
    # otherwise, return_dict_in_generate is applied in the 'result' of each segment in final_segments
    if (
        return_dict_in_generate
        and generation_config.return_dict_in_generate
        and (force_unique_generate_call or not return_timestamps)
    ):
        # only one call to generate_with_fallback, we can return a ModelOutput
        outputs = self._stack_split_outputs(seek_outputs, model_output_type, self.device, kwargs)
        if num_return_sequences > 1:
            if hasattr(outputs, "encoder_attentions") and outputs.encoder_attentions is not None:
                outputs.encoder_attentions = tuple(
                    outputs.encoder_attentions[i][::num_return_sequences]
                    for i in range(len(outputs.encoder_attentions))
                )
            if hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states is not None:
                outputs.encoder_hidden_states = tuple(
                    outputs.encoder_hidden_states[i][::num_return_sequences]
                    for i in range(len(outputs.encoder_hidden_states))
                )
        return outputs

    padded_outputs = generation_whisper._pad_to_max_length(
        current_segments=final_segments,
        pad_token_id=generation_config.pad_token_id,
        device=self.device,
        padding_side="right",
        return_token_timestamps=return_token_timestamps,
        force_unique_generate_call=force_unique_generate_call,
    )

    if return_dict_in_generate and generation_config.return_dict_in_generate:
        generation_whisper.logger.warning_once(
            "You have passed `return_dict_in_generate=True` and `return_timestamps=True`, this automatically sets `return_segments=True` to access the resuls of the underlying calls to GenerationMixin's generate in the returned `segments`."
        )
        return_segments = True
    elif not return_segments and not return_token_timestamps:
        return padded_outputs

    if return_token_timestamps:
        sequences, token_timestamps = padded_outputs
        outputs = {
            "sequences": sequences,
            "token_timestamps": token_timestamps,
        }
    else:
        sequences = padded_outputs
        outputs = {
            "sequences": sequences,
        }

    if return_segments:
        outputs["segments"] = final_segments

    return outputs

@staticmethod
def _prepare_decoder_input_ids(
    cur_bsz,
    init_tokens,
    current_segments,
    batch_idx_map,
    do_condition_on_prev_tokens,
    prompt_ids,
    generation_config,
    config,
    device,
    suppress_tokens,
    timestamp_begin,
    kwargs,
):
    if "decoder_input_ids" in kwargs:
        decoder_input_ids = kwargs.pop("decoder_input_ids")
        return decoder_input_ids, kwargs

    cut_off_length = config.max_target_positions // 2 - 1

    decoder_input_ids = init_tokens[batch_idx_map]

    prev_start_of_text = getattr(generation_config, "prev_sot_token_id", None)
    if prev_start_of_text is None:
        if suppress_tokens is not None and len(suppress_tokens) >= 2:
            prev_start_of_text = suppress_tokens[-2]
        else:
            prev_start_of_text = None

    if any(do_condition_on_prev_tokens) and len(current_segments[0]) > 0:
        # according to https://github.com/openai/whisper/blob/e58f28804528831904c3b6f2c0e473f346223433/whisper/decoding.py#L609
        active_segments = [current_segments[i] if do_condition_on_prev_tokens[i] else None for i in batch_idx_map]

        for segments in active_segments:
            for seg in segments:
                if len(seg["tokens"]) > 2 and seg["tokens"][-2] >= timestamp_begin:
                    # the segment finishes with two timestamp tokens
                    # we need to ignore the last timestamp token
                    # see https://github.com/huggingface/transformers/pull/34537
                    seg["tokens"] = seg["tokens"][:-1]

        if prompt_ids is not None and generation_config.prompt_condition_type == "all-segments":
            prev_ids = prompt_ids
        else:
            one_tensor = torch.ones((cur_bsz, 1), device=device, dtype=torch.long)
            prev_ids = prev_start_of_text * one_tensor[0] if prev_start_of_text is not None else None

        padding = "max_length" if generation_config.cache_implementation == "static" else "longest"
        
        non_zero_segments_ids = []
        for i, seg in enumerate(active_segments):
            tokens_len = sum([len(d["tokens"]) for d in seg])
            if tokens_len > 0:
                non_zero_segments_ids.append(i)
     
        non_zero_segments = [active_segments[idx] for idx in non_zero_segments_ids]
        if len(non_zero_segments) > 0:
            prev_tokens = generation_whisper._pad_to_max_length(
                non_zero_segments,
                generation_config.pad_token_id,
                device=device,
                padding_side="left",
                padding=padding,
                bos_token_tensor=prev_ids,
                cut_off_length=cut_off_length,
            )
            if prev_tokens.shape[0] != decoder_input_ids.shape[0]:
                prev_tokens_out = torch.full((decoder_input_ids.shape[0], prev_tokens.shape[1]), generation_config.pad_token_id, device=device)
                prev_tokens_out[non_zero_segments_ids] = prev_tokens
                prev_tokens = prev_tokens_out.clone()

            decoder_input_ids = torch.cat([prev_tokens, decoder_input_ids], dim=-1)
        

        kwargs["decoder_attention_mask"] = decoder_input_ids != generation_config.pad_token_id
    elif prompt_ids is not None:
        prev_tokens = prompt_ids[None].repeat(decoder_input_ids.shape[0], 1)
        decoder_input_ids = torch.cat([prev_tokens, decoder_input_ids], dim=-1)
        # make sure `"decoder_attention_mask"` is not passed to forward
        kwargs.pop("decoder_attention_mask", None)
    else:
        # make sure `"decoder_attention_mask"` is not passed to forward
        kwargs.pop("decoder_attention_mask", None)

    return decoder_input_ids, kwargs

import random
@staticmethod
def _prepare_segments(prompt_ids, batch_size, generation_config):
    if prompt_ids is not None and generation_config.prompt_condition_type == "first-segment":
        prev_sot_token_id = getattr(generation_config, "prev_sot_token_id", None)

        if isinstance(prompt_ids, list):
            for i in range(len(prompt_ids)):
                if len(prompt_ids[i]) > 0 and prompt_ids[i][0] == prev_sot_token_id: # hf adds in prev text sot during _prepare_decoder_input_ids
                    prompt_ids[i] = prompt_ids[i][1:]
                
            #current_segments = [[{"tokens": torch.tensor([],device='cuda', dtype=torch.long)}] if random.random() < 0.2 else [{"tokens": prompt_ids[i]}] for i in range(batch_size)]
            current_segments = [[{"tokens": prompt_ids[i]}] for i in range(batch_size)]
        else:
            prompt_ids = prompt_ids[1:] if prompt_ids[0] == prev_sot_token_id else prompt_ids
            current_segments = [{"tokens": prompt_ids}] * batch_size

    else:
        current_segments = [[] for _ in range(batch_size)]

    return current_segments

generation_whisper.WhisperGenerationMixin._prepare_segments = _prepare_segments # monkey patch
generation_whisper.WhisperGenerationMixin._prepare_decoder_input_ids = _prepare_decoder_input_ids # monkey patch
generation_whisper.WhisperGenerationMixin.generate = generate # monkey patch


def get_whisper(name='openai/whisper-tiny.en', device=None, dtype=torch.float32, attn_implementation='sdpa'):
    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, attn_implementation=attn_implementation, torch_dtype=dtype)
    if device is not None: model = model.to(device)
    return processor, model

@torch.no_grad()
def generate_with_whisper(
        model:WhisperForConditionalGeneration,
        processor:WhisperProcessor,
        audio:List[np.ndarray],
        device:torch.device,
        dtype=torch.float32,
        initial_prompt:List[str]=None,
        max_batch_size:int=-1,
        return_timestamps:bool=False,
        is_multilingual:bool=False,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        num_beams=5,
        logprob_threshold=-1.0,
        prompt_condition_type='first-segment',
        condition_on_prev_tokens=True,
):  
    if initial_prompt is None: prompt_ids = None
    else: 
        prompt_ids = [torch.tensor(processor.get_prompt_ids(el, return_tensors=None)) for el in initial_prompt]
        print([el.shape for el in prompt_ids])


    transcriptions = []
    processed = 0
    if max_batch_size == -1: max_batch_size = len(audio)

    while processed < len(audio):
        input_features = processor.feature_extractor(
            audio[processed:processed + max_batch_size],
            return_tensors="pt",
            sampling_rate=16_000,
        ).to(device=device, dtype=dtype)


        if prompt_ids is not None:
            cur_prompt_ids = prompt_ids[processed:processed + max_batch_size]
            cur_prompt_ids = [el.to(device=device) for el in cur_prompt_ids]
        else: cur_prompt_ids = None

        predicted_ids = model.generate(
            input_features = input_features['input_features'],
            attention_mask = None,
            prompt_ids = None if cur_prompt_ids is None else cur_prompt_ids,
            return_timestamps=return_timestamps,
            is_multilingual=is_multilingual,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            num_beams=num_beams,
            logprob_threshold=logprob_threshold,
            prompt_condition_type=prompt_condition_type,
            condition_on_prev_tokens=condition_on_prev_tokens,
        )

        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        transcriptions.extend(transcription)
        processed += len(transcription)
        print(f"Processed {processed}/{len(audio)}")


    return transcriptions


from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
      super().__init__()
      self.stops = stops
      self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      stop_count = 0
      for stop in self.stops:
        stop_count = (stop == input_ids[0]).sum().item()

      if stop_count >= self.ENCOUNTERS:
          return True
      return False

@torch.no_grad()
def generate_with_whisper_r1(
        model:WhisperForConditionalGeneration,
        processor:WhisperProcessor,
        audio:List[np.ndarray],
        device:torch.device,
        dtype=torch.float32,
        initial_prompt:List[str]=None,
        return_timestamps:bool=False,
        is_multilingual:bool=False,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        num_beams=5,
        logprob_threshold=-1.0,
        prompt_condition_type='first-segment',
        condition_on_prev_tokens=True,
):     
    max_batch_size = 1
    if initial_prompt is None: prompt_ids = None
    else: prompt_ids = [torch.tensor(processor.get_prompt_ids(el, return_tensors=None)) for el in initial_prompt]

    transcriptions = []
    processed = 0
    if max_batch_size == -1: max_batch_size = len(audio)

    while processed < len(audio):
        input_features = processor.feature_extractor(
            audio[processed:processed + max_batch_size],
            return_tensors="pt",
            sampling_rate=16_000,
        ).to(device=device, dtype=dtype)

        if prompt_ids is not None:
            cur_prompt_ids = prompt_ids[processed:processed + max_batch_size]
            cur_prompt_ids = [el.to(device=device) for el in cur_prompt_ids]
            decoder_input_ids = None
        else: 
            cur_prompt_ids = None
            #decoder_input_ids = torch.tensor(processor.get_prompt_ids("", return_tensors=None)+processor.tokenizer.encode("")[:-1]).to(device).unsqueeze(0)
            decoder_input_ids = torch.tensor(processor.get_prompt_ids("The", return_tensors=None)[:1]).to(device).unsqueeze(0)

        while True:
            predicted_ids = model.generate(
                input_features = input_features['input_features'],
                attention_mask = None,
                prompt_ids = None if cur_prompt_ids is None else cur_prompt_ids,
                return_timestamps=return_timestamps,
                is_multilingual=is_multilingual,
                max_new_tokens=100,
                temperature=0.7,
                compression_ratio_threshold=compression_ratio_threshold,
                num_beams=num_beams,
                logprob_threshold=logprob_threshold,
                prompt_condition_type=prompt_condition_type,
                condition_on_prev_tokens=condition_on_prev_tokens,
                decoder_input_ids=decoder_input_ids,
                return_dict_in_generate=True,
                disable_logits_processor=True,
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[processor.tokenizer.encode("")[1]], encounters=1)]),
            )
            if processor.tokenizer.encode("")[0] in predicted_ids['sequences'][0]:
                break
        #print(predicted_ids['sequences'].shape)

        predicted_ids = model.generate(
            input_features = input_features['input_features'],
            attention_mask = None,
            prompt_ids = None if cur_prompt_ids is None else cur_prompt_ids,
            return_timestamps=return_timestamps,
            is_multilingual=is_multilingual,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            num_beams=num_beams,
            logprob_threshold=logprob_threshold,
            prompt_condition_type=prompt_condition_type,
            condition_on_prev_tokens=condition_on_prev_tokens,
            decoder_input_ids=predicted_ids['sequences'],
            return_dict_in_generate=True,
            disable_logits_processor=True,
        )

        transcription = processor.tokenizer.batch_decode(predicted_ids['sequences'], skip_special_tokens=False)
        print(transcription)
        transcription = processor.tokenizer.batch_decode(predicted_ids['sequences'], skip_special_tokens=True)
        #transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        transcriptions.extend(transcription)
        processed += len(transcription)
        print(f"Processed {processed}/{len(audio)}")


    return transcriptions