import argparse
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from synctxasr.data import dataset_functions
from synctxasr.wer import word_error_rate_detail
import torch
from synctxasr.misc import int_or_none
from synctxasr.asr import get_whisper, generate_with_whisper
import random


def shuffle_text(text:str) -> str:
    words = text.split()
    random.shuffle(words)
    return " ".join(words)

def main(args):
    device = torch.device(args.device)
    
    #processor, model = get_whisper(args.whisper_model, device=device)

    processor, model = get_whisper(args.whisper_model, device=args.device)

    # new_tokens = ["(thinking)", "(.thinking)"]
    # processor.tokenizer.add_tokens(new_tokens)
    # model.resize_token_embeddings(len(processor.tokenizer))

    # checkpoint = 'whisper-tiny-finetuned_1000'
    # # # load the model from the checkpoint
    if args.checkpoint is not None:
        loaded_model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint)
        model.load_state_dict(loaded_model.state_dict(), strict=False)

    dataset_fn = dataset_functions[args.dataset]
    dataset = dataset_fn(args.split)

    step = 0
    finished_recording = [False] * len(dataset)

    if sum(args.indexes) == -1: indexes = list(range(len(dataset))) 
    else: indexes = args.indexes
    print(indexes)

    recordings = [el['process_fn'](el) for i, el in enumerate(dataset) if i in indexes]

    max_steps = max([len(recording) for recording in recordings])

    hyps = {k:"" for k in range(len(recordings))}
    refs = {k:"" for k in range(len(recordings))}
    history = {k:[] for k in range(len(recordings))}
    end_times = {k:0 for k in range(len(recordings))}

    pbar = tqdm(total=max_steps, desc="Processing", unit="step")
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
            start_time, end_time = recording[step]['start'], recording[step]['end']
            if (start_time - end_times[i] > 2.0) and args.reset_history_on_pause:
                history[i] = [] # reset history if a long pause
            end_times[i] = end_time
    
            history_size = len(history[i]) if args.prev_utterances == -1 else min(args.prev_utterances, len(history[i]))
            if args.use_history and args.prev_utterances != 0:
                if args.prev_utterances != 0:
                    initial_prompts.append(" ".join(history[i][-history_size:]) if len(history[i]) > 0 else None)
            else: initial_prompts.append(None)
            indexes.append(i)
            # print(len(initial_prompts[-1])) if initial_prompts[-1] is not None else 0

        if len(cur_audio) == 0: break

        if initial_prompts[0] is None or not args.use_history: initial_prompts = None
        # print(initial_prompts, cur_refs)
        transcription = generate_with_whisper(
            model, 
            processor, 
            cur_audio, 
            device=device, 
            num_beams=args.beam_size,
            initial_prompt=initial_prompts,
            max_batch_size=1,
            return_timestamps=args.return_timestamps,
        )
        #print(transcription)

        for i, idx in enumerate(indexes):
            #print(transcription[i])
            hyps[idx] += transcription[i] + " "
            refs[idx] += cur_refs[i] + " "
            prev_history = transcription[i].strip()
            # if prev_history.endswith('...'):
            #     prev_history = prev_history[:-3]

            if args.shuffle_history:
                prev_history = shuffle_text(prev_history)
                
            if args.strip_fullstop and prev_history.endswith('.'):
                prev_history = prev_history[:-1]
            if args.uppercase_history:
                prev_history = prev_history.upper()
            history[idx].append(prev_history)

        pbar.update(1)

        step += 1
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(list(hyps.values()), list(refs.values()), use_cer=False, normalize=True)
    print(f"Word Error Rate: {wer*100:.4f}%")

    return wer, words, ins_rate, del_rate, sub_rate


    
def calc_wer(words, ins, dels, subs):
    return (ins + dels + subs) / words     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_history', action='store_true')
    parser.add_argument('--beam_size', type=int_or_none, default=5)
    parser.add_argument('--dataset', type=str, default='tedlium3')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--whisper_model', type=str, default='openai/whisper-tiny.en')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--indexes', '-indexes', type=int, nargs='+', help='Indexes of the data to evaluate', default=[-1]) # -1 means all
    parser.add_argument('--sequential', action='store_true', help='Use sequential processing of indexes')
    parser.add_argument('--log_path', type=str, default=None)   
    parser.add_argument('--strip_fullstop', action='store_true', help='Strip fullstop from the history')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--return_timestamps', action='store_true')
    parser.add_argument('--prev_utterances', type=int, default=-1, help='Number of previous utterances to use as context. -1 means all previous utterances.')
    parser.add_argument('--reset_history_on_pause', action='store_true', help='Reset history if there is a long pause between utterances.')
    parser.add_argument('--shuffle_history', action='store_true', help='Shuffle the words in the history to test robustness.')
    parser.add_argument('--uppercase_history', action='store_true', help='Uppercase the history to test robustness.')
    args = parser.parse_args()

    if not torch.cuda.is_available(): args.device = 'cpu' # force cpu if no cuda available

    if not args.sequential:
        main(args)
    else:
        dataset_fn = dataset_functions[args.dataset]
        dataset = dataset_fn(args.split)

        if sum(args.indexes) == -1: indexes = list(range(len(dataset))) 
        else: indexes = args.indexes
        
        total_words = 0
        total_insertions = 0
        total_deletions = 0
        total_substitutions = 0

        for i in indexes:
            print(f"Processing {i}/{len(indexes)}")
            args.indexes = [i]
            wer, words, ins_rate, del_rate, sub_rate = main(args)

            if args.log_path is not None:
                with open(args.log_path, 'a') as f:
                    f.write(f"\n {i}/{len(indexes)}: {wer*100:.4f}%\n")

            total_words += words
            total_insertions += ins_rate * words
            total_deletions += del_rate * words
            total_substitutions += sub_rate * words

            print(f"Current WER: {calc_wer(total_words, total_insertions, total_deletions, total_substitutions)*100:.4f}%")

        print(f"Total WER: {calc_wer(total_words, total_insertions, total_deletions, total_substitutions)*100:.4f}%")

        if args.log_path is not None:
            with open(args.log_path, 'a') as f:
                f.write(f"\n Total WER: {calc_wer(total_words, total_insertions, total_deletions, total_substitutions)*100:.4f}%\n")
