import argparse
from tqdm import tqdm
from synctxasr.data import dataset_functions
from synctxasr.wer import word_error_rate_detail
from synctxasr.misc import int_or_none
import whisper


def main(args):

    asr_model = whisper.load_model(args.whisper_model)

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
        previous_text = None

        for utt_idx, utt in enumerate(tqdm(utterances)):
            waveform = utt['waveform'].squeeze(0)


            #previous_text = None if utt_idx == 0 else utterances[-1]['text'] 
            prev_end_utt = None if utt_idx == 0 else utterances[utt_idx - 1]['end']
            cur_start_utt = utt['start']
            if (prev_end_utt is not None and cur_start_utt - prev_end_utt > 1.0):
                previous_text = None
            
            result = asr_model.transcribe(
                audio=waveform, 
                initial_prompt=previous_text, 
                without_timestamps=True, 
                language='en', 
                task='transcribe',
                beam_size=args.beam_size,
            ) 

            if args.use_history and not args.use_gold_history:
                previous_text = result['text'].strip()
                if previous_text == "": previous_text = None
                elif previous_text[-1] == '.': previous_text = previous_text[:-1] # IMPORTANT!
            elif args.use_gold_history and utt_idx > 0:
                previous_text = utterances[utt_idx - 1]['text'].strip()
                previous_end = utterances[utt_idx - 1]['end']
                current_start = utt['start']
                if previous_text == "": previous_text = None
                if current_start - previous_end > 1.0: previous_text = None # Drop history if there is too much of a gap
            

            cur_hyp += result['text'] + " "
            cur_ref += utt['text'] + " "

        hyps.append(cur_hyp.strip())
        refs.append(cur_ref.strip())

    wer = word_error_rate_detail(hyps, refs, use_cer=False, normalize=True)
    print(wer)
          

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_history', action='store_true')
    parser.add_argument('--use_gold_history', action='store_true')
    parser.add_argument('--beam_size', type=int_or_none, default=5)
    parser.add_argument('--dataset', type=str, default='tedlium3')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--whisper_model', type=str, default='tiny.en')
    parser.add_argument('--indexes', '-indexes', type=int, nargs='+', help='Indexes of the data to evaluate', default=[-1]) # -1 means all

    args = parser.parse_args()

    if args.use_gold_history: args.use_history = True

    main(args)