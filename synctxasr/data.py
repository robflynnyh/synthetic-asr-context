import re
import os
import json
import pickle
from typing import List, Dict
import torchaudio



def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines

def open_txt(path:str) -> str:
    with open(path, 'r') as f:
        return f.read().strip()

def load_json(file_path:str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def convert_str_to_seconds(time_str:str) -> float: # in format: HH:MM:SS convert to seconds
    hours, minutes, seconds = time_str.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def segment_waveform(waveform, frames_per_second, utterances, buffer=0.0):
    for utt in utterances:
        start,end = utt['start'], utt['end']
        start_frame = int(round(start * frames_per_second)) - int(buffer * frames_per_second)
        end_frame = int(round(end * frames_per_second)) + int(buffer * frames_per_second)
        utt['waveform'] = waveform[..., start_frame:end_frame].clone()
    return utterances 


def prepare_chunks(spec, seq_len, overlap):
    spec_n = spec.shape[-1]
    last_ulen, kill_next = None, False

    if spec_n <= seq_len:
        return {0: spec}, [0]

    training_data = {}
    for i in range(0, spec_n, seq_len-overlap):
        audio_chunk = spec[:, :, i:i+seq_len] # [B, C, T]
        u_len = audio_chunk.shape[-1]
        if kill_next:
            break
        elif last_ulen != None and u_len < last_ulen:
            kill_next = True
        last_ulen = u_len
        training_data[i] = audio_chunk
    return training_data, list(training_data.keys())

def gigaspeech_data():
    default_base_path = "/store/store4/data/gigaspeech/"
    default_transcript_file = "GigaSpeech_eval.json" # filtered to only include the test set bcos the full thing is a couple of GB
    splits = ['test'] 

    def relabel_segment_text(segments):
        for seg in segments:
            seg['start'] = seg['begin_time']
            seg['end'] = seg['end_time']
            seg['text'] = seg['text_tn']
            del seg['begin_time']
            del seg['end_time']
            del seg['text_tn']
        return segments

    def process_text_and_audio_fn(rec_dict):
        audio_path = rec_dict['audio']
        audio, sr = torchaudio.load(audio_path)
        assert sr == 16_000, f"Audio sample rate should be 16kHz (got {sr}).."
        length_in_seconds = audio.shape[-1] / sr
        frames_per_second = audio.shape[-1] / length_in_seconds
        segments = relabel_segment_text(rec_dict['text'])
        utterances = segment_waveform(audio, frames_per_second, segments, buffer=0.0)

        return utterances

    def get_text_and_audio(split='test-other', base_path=None, transcript_file=None):
        base_path = base_path or default_base_path
        transcript_file = transcript_file or default_transcript_file
        assert split in splits, f'Split must be in {splits} (got {split})'
        transcript_file_path = os.path.join(base_path, transcript_file)
        transcript_data = load_json(transcript_file_path)
        recordings = transcript_data['audios']
        formatted_split = "{" + split.upper() + "}"
        recordings = [rec for rec in recordings if formatted_split in rec['subsets']]
        assert len(recordings) > 0, f"No recordings found for split {split} in {transcript_file_path}. Something is wrong!"

        return_data = []
        for rec in recordings:
            return_data.append({
                'id': rec['aid'],
                'duration': rec['duration'],
                'source': rec['source'],
                'audio': os.path.join(base_path, rec['path']),
                'text': rec['segments'],
                'process_fn': process_text_and_audio_fn
            })
        return return_data
    
    return get_text_and_audio


def librispeech_data():
    default_base_path = "/store/store4/data/LibriSpeech/original/"
    splits = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']

    def get_utterances(text:List[str], audio_paths:Dict[str, str]):
        utts = []
        for line in text:
            sline = line.split(' ')
            if len(sline) <= 1: continue
            id = sline[0].strip()
            text = ' '.join(sline[1:]).strip().lower()
            audio, sr = torchaudio.load(audio_paths[id])
            assert sr == 16_000, f"Audio sample rate should be 16kHz (got {sr}).."
            utts.append({
                'id': id,
                'text': text,
                'waveform': audio,
                'start': -1,
                'end': -1,
            })
        return utts

    def process_text_and_audio_fn(rec_dict):
        text_path = rec_dict['text']
        text = open_stm(text_path)
        utterances = get_utterances(text, rec_dict['audio'])
        return utterances

    def get_text_and_audio(split='test-other', base_path=None):
        base_path = base_path or default_base_path
    
        assert split in splits, f'Split must be in {splits} (got {split})'
        path = os.path.join(base_path, split)
        
        return_data = []

        for parent_dir in os.listdir(path):
            for child_dir in os.listdir(os.path.join(path, parent_dir)):
                cur_dir = os.path.join(path, parent_dir, child_dir)
                recording_id = f"{parent_dir}-{child_dir}"
                transcript_file = f"{recording_id}.trans.txt"
                transcript_file_path = os.path.join(cur_dir, transcript_file)
                assert os.path.exists(transcript_file_path), f"Transcript file {transcript_file_path} does not exist"
                audio_paths = {file.removesuffix(".flac"): os.path.join(cur_dir, file) for file in os.listdir(cur_dir) if file.endswith('.flac')}
                return_data.append({
                    'id': recording_id,
                    'text': transcript_file_path,
                    'audio': audio_paths,
                    'process_fn': process_text_and_audio_fn
                })
        return return_data

    return get_text_and_audio

 

def tedlium3_segmented_data():
    default_base_path = "/store/store4/data/TEDLIUM_release-3/legacy/"

    def proc_stm_and_timings(stm_path:str):
        stm = open_stm(stm_path)
        utts = []
        for line in stm:
            sline = line.split(' ')
            if len(sline) < 6:
                continue
            a_id, s_id, spk, start, end, meta = sline[:6]
            text = ' '.join(sline[6:])
            if text.strip() == 'ignore_time_segment_in_scoring':
                continue
            text = re.sub(r" '([a-z])", r"'\1", text)
            # remove anything inside angle brackets i.e <...>
            utts.append({'start': float(start), 'end': float(end), 'text': re.sub(r'<[^>]*>', '', text)})
            
        return utts

    def load_tedlium_recording(stm_path:str, sph_path:str):
        utts = proc_stm_and_timings(stm_path)
        audio, sr = torchaudio.load(sph_path)
        length_in_seconds = audio.shape[-1] / sr
        frames_per_second = audio.shape[-1] / length_in_seconds
    
        utterances = segment_waveform(audio, frames_per_second, utts, buffer=0.05)
        return utterances

    def process_text_and_audio_fn(rec_dict):
        utterances = load_tedlium_recording(stm_path=rec_dict['text'], sph_path=rec_dict['audio'])
        return utterances

    def get_text_and_audio(split, base_path=None):
        base_path = base_path or default_base_path
        assert split in ['test', 'dev', 'train'], f'Split must be either test or dev or train(got {split})'
        path = os.path.join(base_path, split)
        recordings = os.listdir(os.path.join(path, 'sph'))

        return_data = []
        for rec in range(len(recordings)):
            return_data.append({
                'id': recordings[rec].replace('.sph', ''),
                'text': os.path.join(path, 'stm', recordings[rec].replace('.sph', '.stm')),
                'audio': os.path.join(path, 'sph', recordings[rec]),
                "process_fn": process_text_and_audio_fn
            })
        return return_data
    return get_text_and_audio


dataset_functions = {
    "tedlium3": tedlium3_segmented_data(),
    "librispeech": librispeech_data(),
    "gigaspeech": gigaspeech_data(),
}


def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)


def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)