from transformers import AutoTokenizer, AutoModelForCausalLM

def create_lm_template(transcription:str):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant. You do not use full stops at the end of your outputs unless absolutely necessary!"}]
        },
        {
            "role": "user",
            "content": [{
                "type": "text", "text": f"Given the following output from my ASR model, predict a plausable sentence that could have come before it. Only ouptut one sentence and nothing else: '{transcription}'."}]
        },
    ]    
    return messages

def create_inputs(transcription:str, tokenizer:AutoTokenizer, device:str):
    messages = create_lm_template(transcription)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    return inputs

def generate_lm_response(transcription:str, model:AutoModelForCausalLM, tokenizer:AutoTokenizer, device:str):
    inputs = create_inputs(transcription, tokenizer, device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id    
        )
        
    outputs = tokenizer.batch_decode(outputs)[0]
    response = outputs.split("<start_of_turn>model\n")[-1].strip().split("<end_of_turn>")[0].strip()
    
    return response