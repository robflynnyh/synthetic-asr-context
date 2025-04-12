import argparse
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from synctxasr.data import dataset_functions
from synctxasr.wer import word_error_rate_detail
import torch

def int_or_none(value):
    if value.lower() == 'none':
        return None
    return int(value)

def main(args):
    device = torch.device(args.device)
    processor = WhisperProcessor.from_pretrained(args.whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model).to(device)

    dataset_fn = dataset_functions[args.dataset]
    dataset = dataset_fn(args.split)

    step = 0
    finished_recording = [False] * len(dataset)

    recordings = [el['process_fn'](el) for el in dataset]

    max_steps = max([len(recording) for recording in recordings])

    hyps = {k:"" for k in range(len(recordings))}
    refs = {k:"" for k in range(len(recordings))}
    history = {k:[] for k in range(len(recordings))}

    while all(finished_recording) == False:
        print(f'Step {step+1}/{max_steps}')
        cur_refs = []
        cur_audio = []
        indexes = []
        initial_prompts = []

        for i, recording in enumerate(recordings):
            if finished_recording[i]: continue
            if step >= len(recording): 
                finished_recording[i] = True
                continue
            cur_refs.append(recording[step]['text'])
         
            cur_audio.append(recording[step]['waveform'].squeeze(0).numpy())
            if args.use_history: initial_prompts.append(history[i][-1] if len(history[i]) > 0 else None)
            else: initial_prompts.append(None)
            indexes.append(i)

        if len(cur_audio) == 0: break

        if initial_prompts[0] is None: initial_prompts = None

        if args.use_history and initial_prompts is not None:
            prompt_ids = [processor.get_prompt_ids(el, return_tensors=None) for el in initial_prompts]
            prompt_ids = processor.tokenizer.pad(
                {"input_ids": prompt_ids},
                padding=True,
                return_tensors="pt"
            )
        else:
            prompt_ids = None

        input_features = processor.feature_extractor(cur_audio, return_tensors="pt", sampling_rate=16_000)
        print(input_features.keys())
        predicted_ids = model.generate(
            input_features = input_features['input_features'].to(device),
            attention_mask = None if prompt_ids is None else prompt_ids['attention_mask'].to(device),
            prompt_ids = None if prompt_ids is None else prompt_ids['input_ids'].to(device),
            return_timestamps=False,
            is_multilingual=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.4,
            #no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            prompt_condition_type='all-segments',
            condition_on_prev_tokens=True,
        )
        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)


        # outputs = model.transcribe(
        #     cur_audio,
        #     lang_codes=['en'] * len(cur_audio),
        #     tasks=['transcribe'] * len(cur_audio),
        #     initial_prompts=initial_prompts,
        #     batch_size=len(cur_audio),
        # )
        # transcription = [" ".join([el['text'] for el in outputs[i]]) for i in range(len(outputs))]
      
               
        for i, idx in enumerate(indexes):
            print(transcription[i])
            hyps[idx] += transcription[i] + " "
            refs[idx] += cur_refs[i] + " "
            prev_history = transcription[i].strip()
            if prev_history.endswith('...'):
                prev_history = prev_history[:-3]
            elif prev_history.endswith('.'):
                prev_history = prev_history[:-1]
            history[idx].append(prev_history)

        step += 1
    wer = word_error_rate_detail(list(hyps.values()), list(refs.values()), use_cer=False, normalize=True)[0]
    print(f"Word Error Rate: {wer*100:.4f}%")


    
          

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_history', action='store_true')
    parser.add_argument('--beam_size', type=int_or_none, default=5)
    parser.add_argument('--dataset', type=str, default='tedlium3')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--whisper_model', type=str, default='openai/whisper-tiny.en')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if not torch.cuda.is_available(): args.device = 'cpu' # force cpu if no cuda available

    main(args)