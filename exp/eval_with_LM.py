import argparse
from tqdm import tqdm
from synctxasr.data import dataset_functions
from synctxasr.wer import word_error_rate_detail
from synctxasr.lm import generate_lm_response
from synctxasr.misc import int_or_none

from transformers import AutoTokenizer, AutoModelForCausalLM
import whisper
import torch


def main(args):
    asr_model = whisper.load_model(args.whisper_model)
    

    lm_model = AutoModelForCausalLM.from_pretrained(
        args.language_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token



    dataset_fn = dataset_functions[args.dataset]
    dataset = dataset_fn(args.split)

    if sum(args.indexes) == -1: indexes = list(range(len(dataset)))
    else: indexes = args.indexes
    
    hyps = []
    refs = []
    
    for i, index in enumerate(indexes):
        recording = dataset[index]
        print(f"Processing {i+1}/{len(indexes)}: {recording['id']}")
        utterances = recording['process_fn'](recording)
        cur_hyp = ""
        cur_ref = ""

        for utt_idx, utt in enumerate(tqdm(utterances)):
            waveform = utt['waveform'].squeeze(0)
            
            initial_result = asr_model.transcribe(
                audio=waveform, 
                initial_prompt=None, 
                without_timestamps=True, 
                language='en', 
                task='transcribe',
                beam_size=args.beam_size,
            ) 
            initial_text = initial_result['text'].strip()
            if initial_text == "":
                cur_ref += utt['text'] + " "
                continue
            
            
            predicted_history = generate_lm_response(initial_text, lm_model, tokenizer, args.device)
       
            if args.strip_fullstop:
                previous_text = predicted_history.strip()
                if previous_text == "": previous_text = None
                elif previous_text[-1] == '.': previous_text = previous_text[:-1]

            second_result = asr_model.transcribe(
                audio=waveform, 
                initial_prompt=predicted_history, 
                without_timestamps=True, 
                language='en', 
                task='transcribe',
                beam_size=args.beam_size,
            ) 

            if args.verbose:
                print(f"Initial: {initial_text}")
                print(f"Predicted History: {predicted_history}")
                print(f"Second Result: {second_result['text']}")
                print(f"Reference: {utt['text']}")
                print('---------------------------------')

            cur_hyp += second_result['text'] + " "
            cur_ref += utt['text'] + " "

        hyps.append(cur_hyp.strip())
        refs.append(cur_ref.strip())

    wer = word_error_rate_detail(hyps, refs, use_cer=False, normalize=True)
    print(wer)
          

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam_size', type=int_or_none, default=5)
    parser.add_argument('--dataset', type=str, default='tedlium3')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--whisper_model', type=str, default='tiny.en')
    parser.add_argument('--language_model', type=str, default='/store/store4/data/huggingface_models/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--strip_fullstop', action='store_true')
    parser.add_argument('--indexes', '-indexes', type=int, nargs='+', help='Indexes of the data to evaluate', default=[-1]) # -1 means all
    parser.add_argument('--verbose', action='store_true', help='Print outputs of each step')


    args = parser.parse_args()
    
    if not torch.cuda.is_available(): args.device = 'cpu'

    main(args)