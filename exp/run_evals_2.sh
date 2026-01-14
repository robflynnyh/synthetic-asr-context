#!/bin/bash

# export PYTHONPATH=$PYTHONPATH:.
# export PYTHONPATH=$PYTHONPATH:../


CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 1 --strip_fullstop --log_path ./results_history/ted_history_nP_1.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 2 --strip_fullstop --log_path ./results_history/ted_history_nP_2.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 3 --strip_fullstop --log_path ./results_history/ted_history_nP_3.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 4 --strip_fullstop --log_path ./results_history/ted_history_nP_4.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 5 --strip_fullstop --log_path ./results_history/ted_history_nP_5.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 6 --strip_fullstop --log_path ./results_history/ted_history_nP_6.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 7 --strip_fullstop --log_path ./results_history/ted_history_nP_7.txt
echo 1

CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history --sequential  --dataset librispeech --split test-other --prev_utterances 1 --strip_fullstop --log_path ./results_history/LS_history_nP_1.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset librispeech --split test-other --prev_utterances 2 --strip_fullstop --log_path ./results_history/LS_history_nP_2.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset librispeech --split test-other --prev_utterances 3 --strip_fullstop --log_path ./results_history/LS_history_nP_3.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset librispeech --split test-other --prev_utterances 4 --strip_fullstop --log_path ./results_history/LS_history_nP_4.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset librispeech --split test-other --prev_utterances 5 --strip_fullstop --log_path ./results_history/LS_history_nP_5.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset librispeech --split test-other --prev_utterances 6 --strip_fullstop --log_path ./results_history/LS_history_nP_6.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset librispeech --split test-other --prev_utterances 7 --strip_fullstop --log_path ./results_history/LS_history_nP_7.txt
echo 2

CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history --sequential  --dataset gigaspeech --split test --prev_utterances 1 --reset_history_on_pause --strip_fullstop --log_path ./results_history/GS_history_rsp_nP_1.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 2 --reset_history_on_pause --strip_fullstop --log_path ./results_history/GS_history_rsp_nP_2.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 3 --reset_history_on_pause --strip_fullstop  --log_path ./results_history/GS_history_rsp_nP_3.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 4 --reset_history_on_pause --strip_fullstop  --log_path ./results_history/GS_history_rsp_nP_4.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 5 --reset_history_on_pause --strip_fullstop  --log_path ./results_history/GS_history_rsp_nP_5.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 6 --reset_history_on_pause --strip_fullstop  --log_path ./results_history/GS_history_rsp_nP_6.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 7 --reset_history_on_pause --strip_fullstop  --log_path ./results_history/GS_history_rsp_nP_7.txt


echo 3
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 6  --log_path ./results_history/GS_history_6.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 7  --log_path ./results_history/GS_history_7.txt

echo 4


CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history --sequential  --dataset gigaspeech --split test --prev_utterances 1  --strip_fullstop --log_path ./results_history/GS_history_nP_1.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 2  --strip_fullstop --log_path ./results_history/GS_history_nP_2.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 3  --strip_fullstop  --log_path ./results_history/GS_history_nP_3.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 4  --strip_fullstop  --log_path ./results_history/GS_history_nP_4.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 5  --strip_fullstop  --log_path ./results_history/GS_history_nP_5.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 6  --strip_fullstop  --log_path ./results_history/GS_history_nP_6.txt
CUDA_VISIBLE_DEVICES="3" python3.10 eval_hf.py --use_history  --sequential --dataset gigaspeech --split test --prev_utterances 7  --strip_fullstop  --log_path ./results_history/GS_history_nP_7.txt











CUDA_VISIBLE_DEVICES="2" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 0 --log_path ./results_history/ted_history_0.txt
CUDA_VISIBLE_DEVICES="2" python3.10 eval_hf.py --use_history --sequential --dataset tedlium3 --split test --prev_utterances 1 --log_path ./results_history/ted_history_1.txt
CUDA_VISIBLE_DEVICES="2" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 2  --log_path ./results_history/ted_history_2.txt
CUDA_VISIBLE_DEVICES="2" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 3  --log_path ./results_history/ted_history_3.txt
CUDA_VISIBLE_DEVICES="2" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 4  --log_path ./results_history/ted_history_4.txt
CUDA_VISIBLE_DEVICES="2" python3.10 eval_hf.py --use_history  --sequential --dataset tedlium3 --split test --prev_utterances 5  --log_path ./results_history/ted_history_5.txt

CUDA_VISIBLE_DEVICES="2" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 1 --reset_history_on_pause --log_path ./results_history/ted_history_rsp_1.txt
CUDA_VISIBLE_DEVICES="2" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 2 --reset_history_on_pause --log_path ./results_history/ted_history_rsp_2.txt
CUDA_VISIBLE_DEVICES="2" python3.10 eval_hf.py --use_history --sequential  --dataset tedlium3 --split test --prev_utterances 3 --reset_history_on_pause --log_path ./results_history/ted_history_rsp_3.txt
CUDA_VISIBLE_DEVICES="2" python3.10 eval_hf.py --use_history  --sequential --dataset tedlium3 --split test --prev_utterances 4 --reset_history_on_pause --log_path ./results_history/ted_history_rsp_4.txt
CUDA_VISIBLE_DEVICES="2" python3.10 eval_hf.py --use_history  --sequential --dataset tedlium3 --split test --prev_utterances 5 --reset_history_on_pause --log_path ./results_history/ted_history_rsp_5.txt



