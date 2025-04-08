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

target_output = 'It’s clear that the speaker was frustrated with their lack of competence and the lack of attention given to their work.'

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
    

print(outputs.shape )
outputs = tokenizer.batch_decode(outputs)
response = outputs[0].split("<start_of_turn>model\n")[-1].strip().split("<end_of_turn>")[0].strip()

print("Model's generated response:", response)

# Now calculate loss for target output
# Create messages with model response role including target output
messages_with_target = messages.copy()
messages_with_target.append({
    "role": "assistant",
    "content": [{"type": "text", "text": target_output}]
})

# Apply chat template for target sequence
inputs_with_target = tokenizer.apply_chat_template(
    messages_with_target,
    tokenize=True,
    return_tensors=None,
).to(model.device)

# Get input length to mask out in loss calculation
prompt_length = inputs.input_ids.shape[1]

# decoded = tokenizer.batch_decode(inputs_with_target[:,prompt_length-1:], skip_special_tokens=False)[0]
# print("Decoded input with target:", decoded)
# exit()

# Set up for loss calculation
with torch.inference_mode():
    # Forward pass
    outputs_with_target = model(inputs_with_target, labels=inputs_with_target)
    
    # Extract loss
    loss = outputs_with_target.loss
    
    # For a more detailed analysis, calculate token-by-token loss
    logits = outputs_with_target.logits[0, :-1, :]  # Shape: [sequence_length-1, vocab_size]
    targets = inputs_with_target[0, 1:]  # Shape: [sequence_length-1]
    
    # Calculate per-token loss using cross entropy (only for tokens after the prompt)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(logits[prompt_length-1:], targets[prompt_length-1:])
    
    # Average loss for target tokens only
    target_loss = token_losses.mean()

print(f"Full sequence loss: {loss.item()}")
print(f"Target output loss: {target_loss.item()}")

# For comparison - calculate perplexity
perplexity = torch.exp(target_loss)
print(f"Perplexity for target output: {perplexity.item()}")