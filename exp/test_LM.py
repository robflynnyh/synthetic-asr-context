from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "/store/store4/data/huggingface_models/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"

device = 'cuda'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map=None
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Given the following output from my ASR model, predict a plausable sentence that could have come before it. Only ouptut one sentence and nothing else: 'buti was so utterly unqualified for this project and so utterly ridic and ignored the brief'."}]
    },
]


inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)


with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        pad_token_id=tokenizer.eos_token_id    
    )
    


outputs = tokenizer.batch_decode(outputs)[0]
response = outputs.split("<start_of_turn>model\n")[-1].strip().split("<end_of_turn>")[0].strip()

print(response)