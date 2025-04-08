import re
import os
from os.path import join
import json

import pickle
from typing import List
import torch, torchaudio
from typing import Tuple



def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines

def open_txt(path:str) -> str:
    with open(path, 'r') as f:
        return f.read().strip()

def convert_str_to_seconds(time_str:str) -> float: # in format: HH:MM:SS convert to seconds
    hours, minutes, seconds = time_str.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def segment_spectrogram(spec, frames_per_second, utterances):
    for utt in utterances:
        start,end = utt['start'], utt['end']
        start_frame = int(round(start * frames_per_second))
        end_frame = int(round(end * frames_per_second))
        utt['spectrogram'] = spec[:, :, start_frame:end_frame].clone()
    return utterances  

def segment_waveform(waveform, frames_per_second, utterances, buffer=0.05):
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
    
        utterances = segment_waveform(audio, frames_per_second, utts)
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
}


def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)


def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)