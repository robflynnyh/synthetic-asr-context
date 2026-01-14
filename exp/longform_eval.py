import argparse
from tqdm import tqdm
from synctxasr.data import dataset_functions
from synctxasr.wer import word_error_rate_detail
from synctxasr.misc import int_or_none
import whisper
import torch
import random
import re



def main(args):

    asr_model = whisper.load_model(args.whisper_model)

    assert args.dataset in dataset_functions, f"Dataset {args.dataset} not found, available datasets: {list(dataset_functions.keys())}"
    dataset_fn = dataset_functions[args.dataset]
    dataset = dataset_fn(args.split)

    if sum(args.indexes) == -1: indexes = list(range(len(dataset)))
    else: indexes = args.indexes
    
    hyps = []
    refs = []
    recording_ids = []
    individual_wers = {}
    
    for i, index in enumerate(indexes):
        recording = dataset[index]
        print(f"Processing {i+1}/{len(indexes)}: {recording['id']}")
        utterances = recording['process_fn'](recording)
        cur_hyp = ""
        cur_ref = ""

        waveform = [[]]
        for utt_idx, utt in enumerate(tqdm(utterances)):
            cur_waveform = utt['waveform'].squeeze(0)

            prev_end_utt = None if utt_idx == 0 else utterances[utt_idx - 1]['end']
            cur_start_utt = utt['start']
            if (prev_end_utt is not None and cur_start_utt - prev_end_utt > 2.0):
                if args.break_on_silence:
                    waveform.append([])

            waveform[-1].append(cur_waveform)
            cur_ref += utt['text'] + " "

        for waveform_chunk in waveform:
            waveform_chunk = torch.cat(waveform_chunk, dim=0)
            result = asr_model.transcribe(
                audio=waveform_chunk,
                condition_on_previous_text=args.use_history,
                without_timestamps=not args.use_timestamps, 
                language='en', 
                task='transcribe',
                beam_size=args.beam_size,
            )
            cur_hyp += result['text'] + " "

        cur_hyp = cur_hyp.strip()
        cur_ref = cur_ref.strip()
        hyps.append(cur_hyp)
        refs.append(cur_ref)
        recording_ids.append(recording['id'])
        cur_wer = word_error_rate_detail([cur_hyp], [cur_ref], use_cer=False, normalize=True)[0]
        print(f"Recording {recording['id']}: WER: {cur_wer*100:.3f}")
        individual_wers[recording['id']] = cur_wer


    wer = word_error_rate_detail(hyps, refs, use_cer=False, normalize=True)[0]
    print(f'Overall WER: {wer*100:.3f}')

    if args.save_path is not None:
        torch.save({
            'hyps': hyps,
            'refs': refs,
            'recording_ids': recording_ids,
            'individual_wers': individual_wers,
            'wer': wer
        }, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_history', action='store_true')

    parser.add_argument('--beam_size', type=int_or_none, default=5)
    parser.add_argument('--use_timestamps', action='store_true')
    parser.add_argument('--dataset', type=str, default='tedlium3')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--whisper_model', type=str, default='tiny.en')
    parser.add_argument('--indexes', '-indexes', type=int, nargs='+', help='Indexes of the data to evaluate', default=[-1]) # -1 means all
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--break_on_silence', action='store_true')

    args = parser.parse_args()


    main(args)