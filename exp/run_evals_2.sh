#!/bin/bash

# export PYTHONPATH=$PYTHONPATH:.
# export PYTHONPATH=$PYTHONPATH:../

DEVICE="1"

CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset tedlium3 --split test --prev_utterances 0 --strip_fullstop --log_path ./results_history/ted_turbo_history_nP_0.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset tedlium3 --split test --prev_utterances 1 --strip_fullstop --log_path ./results_history/ted_turbo_history_nP_1.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset tedlium3 --split test --prev_utterances 2 --strip_fullstop --log_path ./results_history/ted_turbo_history_nP_2.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset tedlium3 --split test --prev_utterances 3 --strip_fullstop --log_path ./results_history/ted_turbo_history_nP_3.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset tedlium3 --split test --prev_utterances 4 --strip_fullstop --log_path ./results_history/ted_turbo_history_nP_4.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset tedlium3 --split test --prev_utterances 5 --strip_fullstop --log_path ./results_history/ted_turbo_history_nP_5.txt   
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset tedlium3 --split test --prev_utterances 6 --strip_fullstop --log_path ./results_history/ted_turbo_history_nP_6.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset tedlium3 --split test --prev_utterances 7 --strip_fullstop --log_path ./results_history/ted_turbo_history_nP_7.txt
echo 0

CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset librispeech --split test-other --prev_utterances 0 --strip_fullstop --log_path ./results_history/LS_turbo_history_nP_0.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset librispeech --split test-other --prev_utterances 1 --strip_fullstop --log_path ./results_history/LS_turbo_history_nP_1.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset librispeech --split test-other --prev_utterances 2 --strip_fullstop --log_path ./results_history/LS_turbo_history_nP_2.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset librispeech --split test-other --prev_utterances 3 --strip_fullstop --log_path ./results_history/LS_turbo_history_nP_3.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset librispeech --split test-other --prev_utterances 4 --strip_fullstop --log_path ./results_history/LS_turbo_history_nP_4.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset librispeech --split test-other --prev_utterances 5 --strip_fullstop --log_path ./results_history/LS_turbo_history_nP_5.txt   
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset librispeech --split test-other --prev_utterances 6 --strip_fullstop --log_path ./results_history/LS_turbo_history_nP_6.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset librispeech --split test-other --prev_utterances 7 --strip_fullstop --log_path ./results_history/LS_turbo_history_nP_7.txt
echo 1

CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 0 --reset_history_on_pause --strip_fullstop --log_path ./results_history/GS_turbo_history_rsp_nP_0.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 1 --reset_history_on_pause --strip_fullstop --log_path ./results_history/GS_turbo_history_rsp_nP_1.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 2 --reset_history_on_pause --strip_fullstop --log_path ./results_history/GS_turbo_history_rsp_nP_2.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 3 --reset_history_on_pause --strip_fullstop --log_path ./results_history/GS_turbo_history_rsp_nP_3.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 4 --reset_history_on_pause --strip_fullstop --log_path ./results_history/GS_turbo_history_rsp_nP_4.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 5 --reset_history_on_pause --strip_fullstop --log_path ./results_history/GS_turbo_history_rsp_nP_5.txt   
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 6 --reset_history_on_pause --strip_fullstop --log_path ./results_history/GS_turbo_history_rsp_nP_6.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 7 --reset_history_on_pause --strip_fullstop --log_path ./results_history/GS_turbo_history_rsp_nP_7.txt
echo 2

CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 0 --strip_fullstop  --log_path ./results_history/GS_turbo_history_nP_0.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 1 --strip_fullstop  --log_path ./results_history/GS_turbo_history_nP_1.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 2 --strip_fullstop  --log_path ./results_history/GS_turbo_history_nP_2.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 3 --strip_fullstop  --log_path ./results_history/GS_turbo_history_nP_3.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 4 --strip_fullstop  --log_path ./results_history/GS_turbo_history_nP_4.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 5 --strip_fullstop  --log_path ./results_history/GS_turbo_history_nP_5.txt   
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 6 --strip_fullstop  --log_path ./results_history/GS_turbo_history_nP_6.txt
CUDA_VISIBLE_DEVICES="$DEVICE" python3.10 eval_hf.py --use_history --sequential --whisper_model openai/whisper-large-v3-turbo  --dataset gigaspeech --split test --prev_utterances 7 --strip_fullstop  --log_path ./results_history/GS_turbo_history_nP_7.txt





