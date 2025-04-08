import argparse
from tqdm import tqdm
from transformers import AutoProcessor, WhisperForConditionalGeneration
from synctxasr.data import dataset_functions
from synctxasr.wer import word_error_rate_detail
import torch

def int_or_none(value):
    if value.lower() == 'none':
        return None
    return int(value)

def main(args):

    processor = AutoProcessor.from_pretrained(args.whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model)
    model.to(args.device)


    dataset_fn = dataset_functions[args.dataset]
    dataset = dataset_fn(args.split)

    step = 0
    finished_recording = [False] * len(dataset)

    recordings = [el['process_fn'](el) for el in dataset]

    max_steps = max([len(recording) for recording in recordings])

    hyps = {k:"" for k in range(len(recordings))}
    refs = {k:"" for k in range(len(recordings))}

    while all(finished_recording) == False:
        cur_refs = []
        cur_audio = []
        indexes = []

        for i, recording in enumerate(recordings):
            if finished_recording[i]: continue
            if step >= len(recording): 
                finished_recording[i] = True
                continue
            cur_refs.append(recording[step]['text'])
            cur_audio.append(recording[step]['waveform'].squeeze(0).numpy())
            indexes.append(i)
        if len(cur_audio) == 0: break

        print([el.shape[0]/16000 for el in cur_audio])
        inputs = processor(cur_audio, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=16_000)
        inputs = inputs.to(args.device, torch.float32)
        
        generated_ids = model.generate(
            **inputs, 
            return_timestamps=False,
            do_sample=True, 
            num_beams=args.beam_size,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            logprob_threshold=-1.0,
        )
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        for i, idx in enumerate(indexes):
            print(transcription[i])
            hyps[idx] += transcription[i] + " "
            refs[idx] += cur_refs[i] + " "

        step += 1
    wer = word_error_rate_detail(list(hyps.values()), list(refs.values()), use_cer=False, normalize=True)[0]
    print(f"Word Error Rate: {wer*100:.4f}%")


    
          

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_history', action='store_true')
    parser.add_argument('--beam_size', type=int_or_none, default=5)
    parser.add_argument('--dataset', type=str, default='tedlium3')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--whisper_model', type=str, default='/store/store4/data/huggingface_models/models--openai--whisper-tiny.en/snapshots/87c7102498dcde7456f24cfd30239ca606ed9063')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if not torch.cuda.is_available(): args.device = 'cpu' # force cpu if no cuda available

    main(args)