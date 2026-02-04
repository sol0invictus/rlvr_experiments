
import torch
from transformers import AutoTokenizer
from latent_qwen import LatentQwen2ForCausalLM

def test_latent_forward():
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    num_latent = 4
    
    print(f"Loading model {model_name} with {num_latent} latent thoughts...")
    model = LatentQwen2ForCausalLM.from_pretrained(
        model_name, 
        num_latent_thoughts=num_latent,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    model.set_special_token_ids(think_id)
    
    # Construct input
    text = "User: Hello <think> thought </think> Answer: Hi"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print(f"Input text: {text}")
    print(f"Input IDs: {inputs.input_ids}")
    print(f"Think ID: {think_id}")
    
    # Forward pass
    outputs = model(**inputs)
    
    logits = outputs.logits
    print(f"Logits shape: {logits.shape}")
    
    # Expected shape:
    # Original length L
    # We insert num_latent steps.
    # So expected L + num_latent.
    
    expected_len = inputs.input_ids.shape[1] + num_latent
    if logits.shape[1] == expected_len:
        print("SUCCESS: Logits shape matches expected length (Original + Latent).")
    else:
        print(f"FAILURE: Logits shape {logits.shape[1]} != Expected {expected_len}")

    # Check if gradients would flow (requires grad)
    # This is inference mode usually unless we train()
    # model.train()
    # out = model(**inputs)
    # loss = out.logits.sum()
    # loss.backward()
    # print("Backward pass works.")

if __name__ == "__main__":
    test_latent_forward()
