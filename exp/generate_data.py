import argparse
from tqdm import tqdm
from synctxasr.data import dataset_functions
from synctxasr.misc import int_or_none
import whisper
import torch
import os

def save_utterance(utt, save_path):
    """Save the utterance to the specified path."""
    full_save_path = os.path.join(save_path, f"{utt['rec_id']}_{utt['utt_idx']}.pt")
    torch.save(utt, full_save_path)

def main(args):

    asr_model = whisper.load_model(args.whisper_model)

    dataset_fn = dataset_functions[args.dataset]
    dataset = dataset_fn(args.split)

    if sum(args.indexes) == -1: indexes = list(range(len(dataset)))
    else: indexes = args.indexes
    
    for i, index in enumerate(indexes):
        recording = dataset[index]
        print(f"Processing {i+1}/{len(indexes)}: {recording['id']}")
        utterances = recording['process_fn'](recording)

        for utt_idx, utt in enumerate(tqdm(utterances)):
            waveform = utt['waveform'].squeeze(0)
            
            result = asr_model.transcribe(
                audio=waveform, 
                initial_prompt=None, 
                without_timestamps=True, 
                language='en', 
                task='transcribe',
                beam_size=args.beam_size,
            ) 

            utt['generation'] = result['text'].strip()
            utt['utt_idx'] = utt_idx
            utt['rec_id'] = recording['id']
            
            if utt_idx == 0:
                utt['previous_generation'] = None
                utt['previous_gold_text'] = None
            elif (utt['start'] - utterances[utt_idx - 1]['end']) > 1.0:
                utt['previous_generation'] = None
                utt['previous_gold_text'] = None
            else:
                utt['previous_generation'] = utterances[utt_idx - 1]['generation'].strip()
                utt['previous_gold_text'] = utterances[utt_idx - 1]['text'].strip()

            # Save the utterance
            save_utterance(utt, args.save_path)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam_size', type=int_or_none, default=5)
    parser.add_argument('--dataset', type=str, default='tedlium3')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--whisper_model', type=str, default='tiny.en')
    parser.add_argument('--indexes', '-indexes', type=int, nargs='+', help='Indexes of the data to generate data for', default=[-1]) # -1 means all
    parser.add_argument('--save_path', type=str, default='/store/store4/data/TEDLIUM3_Whisper_tiny_en_outputs/')

    args = parser.parse_args()

    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    args.save_path = os.path.join(args.save_path, args.split)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    
    
    main(args)